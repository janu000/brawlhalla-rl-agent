import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical

import pickle
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from torch.utils.tensorboard.writer import SummaryWriter

from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
from Policy import ActionEmbeddingRecurrentPolicy

from BrawlhallaEnv import BrawlhallaEnv
from config import config


class ExpertEpisodeDataset(Dataset):
    def __init__(self, data_dir='data/'):
        self.episodes = self._load_all_episodes(data_dir)
        if not self.episodes:
            raise ValueError(f"No expert episodes found in {data_dir}")

    def _load_all_episodes(self, data_dir):
        episodes_list = []
        episode_files = []

        for filename in os.listdir(data_dir):
            if filename.startswith('expert_episode_') and filename.endswith('.pkl'):
                try:
                    episode_num = int(filename.split('_')[2].split('.')[0])
                    episode_files.append((episode_num, os.path.join(data_dir, filename)))
                except ValueError:
                    print(f"Warning: Skipping malformed episode file: {filename}")
                    continue
        
        episode_files.sort()

        for episode_num, filepath in episode_files:
            try:
                with open(filepath, 'rb') as f:
                    episode_data = pickle.load(f)
                    episodes_list.append(episode_data)
            except Exception as e:
                print(f"Error loading episode from {filepath}: {e}")
                continue
        return episodes_list

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        # Returns a single episode dictionary
        episode = self.episodes[idx]
        
        # Convert NumPy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(episode["obs"]["image"]).float() # Images are usually float for models
        last_executed_action_tensor = torch.from_numpy(episode["obs"]["last_executed_action"]).long() # Action indices are long
        actions_tensor = torch.from_numpy(episode["actions"]).long()
        dones_tensor = torch.from_numpy(episode["dones"]).bool()

        return {
            "obs": {
                "image": image_tensor,
                "last_executed_action": last_executed_action_tensor
            },
            "actions": actions_tensor,
            "dones": dones_tensor
        }

def collate_fn(batch):
    # batch is a list of dictionaries, each representing an episode

    max_T = max(episode["actions"].shape[0] for episode in batch)

    batched_images = []
    batched_last_executed_actions = []
    batched_actions = []
    batched_dones = []
    
    for episode in batch:
        T = episode["actions"].shape[0]

        # Padding images
        image = episode["obs"]["image"]
        padded_image = torch.zeros(max_T, *image.shape[1:], dtype=image.dtype)
        padded_image[:T] = image
        batched_images.append(padded_image)

        # Padding last_executed_action
        last_executed_action = episode["obs"]["last_executed_action"]
        padded_last_executed_action = torch.zeros(max_T, *last_executed_action.shape[1:], dtype=last_executed_action.dtype)
        padded_last_executed_action[:T] = last_executed_action
        batched_last_executed_actions.append(padded_last_executed_action)

        # Padding actions
        actions = episode["actions"]
        padded_actions = torch.zeros(max_T, dtype=actions.dtype) 
        padded_actions[:T] = actions
        batched_actions.append(padded_actions)

        # Padding dones
        dones = episode["dones"]
        padded_dones = torch.zeros(max_T, dtype=dones.dtype) 
        padded_dones[:T] = dones
        batched_dones.append(padded_dones)

    return {
        "obs": {
            "image": torch.stack(batched_images),
            "last_executed_action": torch.stack(batched_last_executed_actions)
        },
        "actions": torch.stack(batched_actions),
        "dones": torch.stack(batched_dones)
    }

def create_valid_mask(dones):
    """
    dones: [B, T] tensor of 0/1 indicating episode ends.

    Returns mask: [B, T], 1 for valid timesteps, 0 for padding/after-done.
    """
    B, T = dones.shape
    device = dones.device

    # Find the index of the first '1' (done) for each batch element.
    # Add a '1' at the end of each sequence to ensure argmax always finds a '1'.
    dones_extended = torch.cat((dones, torch.ones(B, 1, dtype=dones.dtype, device=device)), dim=1)
    
    end_indices = (dones_extended == 1).int().argmax(dim=1)
    
    timesteps = torch.arange(T, device=device).unsqueeze(0) # [1, T]
    
    mask = (timesteps <= end_indices.unsqueeze(1)).float()
    
    return mask

def train_recurrent_bc(model, dataloader, epochs=10, lr=1e-4):
    model.policy.train()
    device = model.device
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    writer = SummaryWriter(log_dir="./runs/bc_pretrain")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            obs_batch = {k: v.to(device) for k, v in batch["obs"].items()}
            action_batch = batch["actions"].to(device)      # [B, T]
            done_batch = batch["dones"].to(device)          # [B, T], bool or int

            B, T = action_batch.shape
            
            # Initial episode_starts for t=0: all episodes are starting
            episode_starts = torch.ones(B, dtype=torch.float, device=device)
            
            # Get the full RNNStates namedtuple, then extract the actor's (h,c) tuple for the forward pass
            lstm_states = model.policy.initial_state(batch_size=B)
            
            # Create valid mask for padded timesteps and timesteps after episode end
            valid_mask = create_valid_mask(done_batch)  # [B, T], float

            total_loss = 0.0
            for t in range(T):
                obs_t = {k: v[:, t] for k, v in obs_batch.items()}  # [B, ...]
                action_t = action_batch[:, t]                         # [B]
                done_t = done_batch[:, t].float()                     # [B]

                # forward pass
                action_dist, lstm_states = model.policy.get_distribution(obs_t, lstm_states, episode_starts)

                # calculate log prob
                log_prob = action_dist.log_prob(action_t)

                # Calculate loss for this timestep and apply valid mask
                step_loss = -log_prob
                step_loss *= valid_mask[:, t]

                # Accumulate loss
                total_loss += step_loss.sum() # Sum across batch

                # Update episode_starts for done episodes: 0 for done, 1 for not done
                episode_starts = (1.0 - done_t)

            # Average the total loss by the number of valid elements in the batch
            total_valid = valid_mask.sum()
            final_loss = total_loss / total_valid if total_valid > 0 else torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            epoch_loss += final_loss.item()

            # Log to TensorBoard
            if writer:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("BC_Loss/train", final_loss.item(), global_step)

        print(f"[Epoch {epoch + 1}] Avg Loss: {epoch_loss / len(dataloader):.6f}")
    
    writer.close()

if __name__ == "__main__":

    dataset = ExpertEpisodeDataset(data_dir='data/')
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )

    print(f"Loaded {len(dataset)} episodes.")
    print(f"Total frames across all episodes: {sum(e['actions'].shape[0] for e in dataset.episodes)}")


    env = BrawlhallaEnv(config=config)
    env.observe_only = True

    policy_kwargs = dict(
        n_lstm_layers=1,
        lstm_hidden_size=config["LSTM_HIDDEN_SIZE"],
        shared_lstm=True,  # share between actor and critic
        enable_critic_lstm=False,    # Disable critic-only LSTM
        features_extractor_kwargs=dict(
            num_actions=env.num_actions,
            in_channel=env.img_shape[0],
            act_emb_dim=config["EMBED_DIM"],
        ),
    )

    checkpoint_callback = CheckpointCallback(
    save_freq=20_000,  # Save every N steps
    save_path='./checkpoints/',
    name_prefix='ppo_brawlhalla'
    )

    model = RecurrentPPO(
        policy=ActionEmbeddingRecurrentPolicy,
        env=env,
        verbose=1,
        tensorboard_log="./ppo_brawlhalla_logs",
        policy_kwargs=policy_kwargs,
        learning_rate=config["LEARNING_RATE"],
        n_steps=config["N_STEPS"],
        batch_size=config["BATCH_SIZE"],
        n_epochs=config["N_EPOCHS"],
        use_sde=False,  # required for recurrent models
    )

    try:
        # Train using behavior cloning on expert data
        train_recurrent_bc(model, dataloader, epochs=10, lr=1e-4)
        
        model.save("checkpoints/bc_pretrained_model")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Closing environment...")
    finally:
        env.close()