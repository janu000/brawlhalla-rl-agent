import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import *

class BrawlhallaEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8),
            'action_history': spaces.Box(low=0, high=NUM_ACTIONS - 1, shape=(HISTORY_LENGTH,), dtype=np.int64)
        })

        self.action_space = spaces.Discrete(NUM_ACTIONS)


    def reset(self, seed=0):
        # Reset game or restart episode
        obs = self._get_obs()

        info = {}
        return obs, info

    def step(self, action):
        self.perform_action(action)

        # Wait a few frames or use a fixed delay
        # Read next observation
        obs = self._get_obs()

        reward = self._compute_reward(action)
        terminated = self.check_done()
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        image = np.random.randint(0, 256, size=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        action_history = np.random.randint(0, NUM_ACTIONS, size=(HISTORY_LENGTH,), dtype=np.int64)  # e.g., last N actions
        return {"image": image, "action_history": action_history}

    def perform_action(self, action):
        # Send keypress using pynput or pyautogui
        pass

    def _compute_reward(self, action_idx):
        
        return 1 - action_idx/NUM_ACTIONS  # placeholder

    def check_done(self):
        # Detect if game is over
        return False
