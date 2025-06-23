"""
Behavior Cloning Pretraining Script for Brawlhalla RL Agent
Loads expert demonstration data, trains a recurrent policy using behavior cloning, and saves the pretrained model.
"""
import pickle
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from tqdm import tqdm

from BrawlhallaEnv import BrawlhallaEnv
from config import config


class ExpertEpisodeDataset(Dataset):
    """
    PyTorch Dataset for loading expert demonstration episodes from disk.
    Each item is a full episode (sequence of images, actions, and dones).
    Supports both lazy loading and pre-loading all data into memory.
    """
    def __init__(self, data_dir='data/', lazy_loading=True):
        self.lazy_loading = lazy_loading
        if lazy_loading:
            self.episode_files = []
            for filename in os.listdir(data_dir):
                if filename.startswith('expert_episode_') and filename.endswith('.pkl'):
                    try:
                        episode_num = int(filename.split('_')[2].split('.')[0])
                        self.episode_files.append((episode_num, os.path.join(data_dir, filename)))
                    except ValueError:
                        print(f"Warning: Skipping malformed episode file: {filename}")
                        continue
            self.episode_files.sort()
        else:
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
        if self.lazy_loading:
            return len(self.episode_files)
        else:
            return len(self.episodes)

    def __getitem__(self, idx):
        if self.lazy_loading:
            return self._getitem_lazy(idx)
        else:
            return self._getitem_preloaded(idx)

    def _getitem_lazy(self, idx):
        """
        Load episode from disk on demand (lazy loading).
        """
        _, filepath = self.episode_files[idx]
        with open(filepath, 'rb') as f:
            episode = pickle.load(f)
        image_tensor = torch.from_numpy(episode["obs"]).float()
        actions_tensor = torch.from_numpy(episode["actions"]).long()
        dones_tensor = torch.from_numpy(episode["dones"]).bool()
        return {
            "obs": image_tensor,
            "actions": actions_tensor,
            "dones": dones_tensor
        }

    def _getitem_preloaded(self, idx):
        """
        Return episode from pre-loaded data in memory.
        """
        episode = self.episodes[idx]
        image_tensor = torch.from_numpy(episode["obs"]).float()
        actions_tensor = torch.from_numpy(episode["actions"]).long()
        dones_tensor = torch.from_numpy(episode["dones"]).bool()
        return {
            "obs": image_tensor,
            "actions": actions_tensor,
            "dones": dones_tensor
        }

def collate_fn(batch):
    """
    Collate function for batching variable-length episodes with padding.
    """
    lengths = [episode["actions"].shape[0] for episode in batch]
    if all(l == lengths[0] for l in lengths):
        # All episodes have the same length, no need to pad
        images = torch.stack([episode["obs"] for episode in batch])
        actions = torch.stack([episode["actions"] for episode in batch])
        dones = torch.stack([episode["dones"] for episode in batch])
        return {
            "obs": images,
            "actions": actions,
            "dones": dones
        }
    
    # Otherwise, pad as before
    max_T = max(lengths)
    batched_images = []
    batched_actions = []
    batched_dones = []
    
    for episode in batch:
        T = episode["actions"].shape[0]

        # Padding images
        image = episode["obs"]
        padded_image = torch.zeros(max_T, *image.shape[1:], dtype=image.dtype)
        padded_image[:T] = image
        batched_images.append(padded_image)

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
        "obs": torch.stack(batched_images),
        "actions": torch.stack(batched_actions),
        "dones": torch.stack(batched_dones)
    }

def create_valid_mask(dones):
    """
    Create a mask for valid (non-padding, non-after-done) timesteps in a batch.
    Args:
        dones: [B, T] tensor of 0/1 indicating episode ends.
    Returns:
        mask: [B, T], 1 for valid timesteps, 0 for padding/after-done.
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

def initial_state(self, batch_size: int):
    """
    Initialises the LSTM hidden and cell states.
    """
    lstm_hidden_size = self.lstm_hidden_state_shape[-1] 
    n_lstm_layers = self.lstm_hidden_state_shape[0]

    hidden_state = torch.zeros(n_lstm_layers, batch_size, lstm_hidden_size, device=self.device)
    cell_state = torch.zeros(n_lstm_layers, batch_size, lstm_hidden_size, device=self.device)
    
    return (hidden_state, cell_state)

RecurrentActorCriticCnnPolicy.initial_state = initial_state

def train_recurrent_bc(model, dataloader, epochs=10, lr=1e-4, alpha=1e-2, writer=None, checkpoint_dir=None, checkpoint_freq=None):
    """
    Train a recurrent policy using behavior cloning on expert data.
    Logs loss to TensorBoard and saves checkpoints periodically.
    """
    model.policy.train()
    device = model.device
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} Batches"):
            obs_batch = batch["obs"].to(device)
            action_batch = batch["actions"].to(device)
            done_batch = batch["dones"].to(device)

            B, T = action_batch.shape

            episode_starts = torch.ones(B, dtype=torch.float, device=device)
            lstm_states = model.policy.initial_state(batch_size=B)
            prev_action = torch.zeros(B, dtype=torch.float, device=device)

            valid_mask = create_valid_mask(done_batch)
            total_log_loss = 0.0
            total_entropy_loss = 0.0
            total_loss = 0.0

            for t in range(T):
                obs_t = obs_batch[:, t]
                action_t = action_batch[:, t]
                done_t = done_batch[:, t].float()

                # forward pass
                action_dist, lstm_states = model.policy.get_distribution(obs_t, lstm_states, episode_starts)

                log_prob = action_dist.log_prob(action_t)
                entropy = action_dist.entropy()

                # reduce importancy of holding a key vs pressing a key
                action_held_weight = 1 - (action_t == prev_action).float() * 0.9

                step_log_loss = -log_prob * valid_mask[:, t]
                step_entropy_loss = -entropy * valid_mask[:, t]
                step_loss = action_held_weight * step_log_loss + alpha * step_entropy_loss

                total_log_loss += step_log_loss.sum()
                total_entropy_loss += step_entropy_loss.sum()
                total_loss += step_loss.sum()


                # Update episode_starts for done episodes: 0 for done, 1 for not done
                episode_starts = (1.0 - done_t)
                prev_action = action_t

                global_step += B
                
            total_valid = valid_mask.sum()

            avg_log_loss = total_log_loss / total_valid if total_valid > 0 else torch.tensor(0.0, device=device)
            avg_entropy_loss = total_entropy_loss / total_valid if total_valid > 0 else torch.tensor(0.0, device=device)
            final_loss = total_loss / total_valid if total_valid > 0 else torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            epoch_loss += final_loss.item()

            # Log to TensorBoard
            if writer:
                writer.add_scalar("BC_Loss/train", final_loss.item(), global_step + 1)
                writer.add_scalar("LogLikelihood_Loss/train", avg_log_loss.item(), global_step + 1)
                writer.add_scalar("Entropy_Loss/train", avg_entropy_loss.item(), global_step + 1)

        # Checkpoint saving
        if checkpoint_dir and checkpoint_freq and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"bc_pretrain_step_{global_step}.zip")
            model.save(checkpoint_path)
            print(f"[Checkpoint] Saved model at step {global_step} to {checkpoint_path}")

        print(f"[Epoch {epoch + 1}] Avg Loss: {epoch_loss / len(dataloader):.6f}")
    
    if writer:
        writer.close()

if __name__ == "__main__":
    """
    Main entry point: loads expert data, sets up environment/model, and runs BC training.
    """
    dataset = ExpertEpisodeDataset(data_dir='data/', lazy_loading=False)
    print(f"Loaded {len(dataset)} episodes.")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )

    env = BrawlhallaEnv(config=config, observe_only=True)

    policy_kwargs = dict(
        n_lstm_layers=1,
        lstm_hidden_size=config["LSTM_HIDDEN_SIZE"],
        shared_lstm=True,  # share between actor and critic
        enable_critic_lstm=False,    # Disable critic-only LSTM,
    )

    model = RecurrentPPO(
        policy=RecurrentActorCriticCnnPolicy,
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
        train_recurrent_bc(
            model, dataloader, epochs=2, lr=1e-4, alpha=1e-2,
            writer=SummaryWriter(log_dir="./training_logs/bc_pretrain"),
            checkpoint_dir="./checkpoints/",
            checkpoint_freq=10 # in epochs
        )
        model.save("checkpoints/bc_pretrained_model")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Closing environment...")
    finally:
        env.close()