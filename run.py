import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sb3_contrib import RecurrentPPO
from BrawlhallaEnv import BrawlhallaEnv
from config import config

env = BrawlhallaEnv(config=config)
policy_kwargs = dict(
    n_lstm_layers=1,
    lstm_hidden_size=config["LSTM_HIDDEN_SIZE"],
    shared_lstm=True,
    enable_critic_lstm=False,
)

model_path = "checkpoints/bc_pretrained_model.zip"
model = RecurrentPPO.load(model_path, env=env, policy_kwargs=policy_kwargs)

def batch_obs(obs):
    return {k: np.expand_dims(v, axis=0) for k, v in obs.items()}

obs, _ = env.reset()
obs_batched = batch_obs(obs)

lstm_states = None
episode_starts = True

for _ in range(1000):  # Run for 1000 steps or until done
    action, lstm_states = model.predict(obs_batched, state=lstm_states, episode_start=episode_starts, deterministic=False)
    action = int(action[0])
    print(env.controller.ACTION_LUT[action])

    obs, reward, termniated, _, _ = env.step(action)
    obs_batched = batch_obs(obs)

    
    episode_starts = termniated
    if termniated:
        obs = env.reset()
        lstm_states = None
        episode_starts = True

env.close()