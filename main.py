import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
import time

from BrawlhallaEnv import BrawlhallaEnv
from Policy import ActionEmbeddingRecurrentPolicy
from config import config

env = BrawlhallaEnv(config=config)

policy_kwargs = dict(
    n_lstm_layers=1,
    lstm_hidden_size=config["LSTM_HIDDEN_SIZE"],
    shared_lstm=True,  # share between actor and critic
    enable_critic_lstm=False,    # Disable critic-only LSTM
    features_extractor_kwargs=dict(
        num_actions=env.num_actions,
        in_channel=config["IMAGE_CHANNELS"],
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
    tensorboard_log="./training_logs/RecurrentPPO",
    policy_kwargs=policy_kwargs,
    learning_rate=config["LEARNING_RATE"],
    n_steps=config["N_STEPS"],
    batch_size=config["BATCH_SIZE"],
    n_epochs=config["N_EPOCHS"],
    use_sde=False,  # required for recurrent models
)

print("Enter Brawhalla within 2s")
time.sleep(2)
print("Starting Training")

try:
    model.learn(total_timesteps=400_000, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Closing environment...")
finally:
    env.close()