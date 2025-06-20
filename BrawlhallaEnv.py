import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import atexit

from config import *
from ScreenRecorder import ScreenRecorder
import controller

DMG_COLOR_LUT = []
with open("healthcolors.txt", 'r') as f:
    for line in f:
        # Remove brackets and split into integers
        line = line.strip().replace('[', '').replace(']', '')
        if line:
            parts = list(map(int, line.split()))
            if len(parts) == 3:
                DMG_COLOR_LUT.append(tuple(parts))

class BrawlhallaEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8),
            'action_history': spaces.Box(low=0, high=NUM_ACTIONS - 1, shape=(HISTORY_LENGTH,), dtype=np.int64)
        })

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.recorder = ScreenRecorder(fps=REC_FPS, region=None, buffer_size=10, render=False, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT))
        self.recorder.start()
        time.sleep(.1)

        atexit.register(self.close)

        self.step_time = None

        self.player_healthtracker = HealthTracker(dmg_color_lut=DMG_COLOR_LUT)
        self.opponent_healthtracker = HealthTracker(dmg_color_lut=DMG_COLOR_LUT)

        self.last_death_time = 0

    def close(self):
        if self.recorder is not None:
            self.recorder.stop()
            controller.release_keys(controller.get_active_keys())
            cv2.destroyAllWindows()

    def reset(self, seed=0):
        # Reset game or restart episode
        obs = self._get_obs()

        info = {}
        return obs, info

    def step(self, action_idx):

        # self.perform_action(action_idx)

        # Enforce LEARNING_FPS
        current_time = time.monotonic()
        if self.step_time:
            time_per_frame = 1.0 / LEARNING_FPS
            elapsed_time = current_time - self.step_time
            sleep_duration = time_per_frame - elapsed_time
        else:
            sleep_duration = 0.5 / LEARNING_FPS

        if sleep_duration > 0:
            time.sleep(sleep_duration)

        self.step_time = current_time

        # Read next observation
        obs = self._get_obs()

        reward = self._compute_reward(action_idx)
        terminated = self.check_done()
        truncated = False
        info = {}       

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
    
        frame = self.recorder.get_latest_frame()
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        if IMAGE_CHANNELS == 1: 
            image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)[np.newaxis, :, :]
        else:
            transposed_frame = np.transpose(resized_frame, (2, 0, 1))
            image = transposed_frame

        expected_shape = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        if image.shape != expected_shape:
            raise ValueError(
                f"Image shape mismatch. Expected {expected_shape}, but got {image.shape}"
            )

        action_history = np.random.randint(0, NUM_ACTIONS, size=(HISTORY_LENGTH,), dtype=np.int64)
        return {"image": image, "action_history": action_history}

    def perform_action(self, action_idx):
        keys_to_press = controller.ACTION_MAPPER.get(action_idx, [])
        active_keys = set(controller.get_active_keys())

        keys_to_press_set = set(keys_to_press)

        # Release keys that are currently active but not needed anymore
        keys_to_release = active_keys - keys_to_press_set
        controller.release_keys(list(keys_to_release))

        # Press new keys
        controller.press_keys(keys_to_press)

    def _compute_reward(self, action_idx):
        frame = self.recorder.get_latest_frame()
        
        # get colors at the position of the respective healthbars
        player_health_bgr = frame[HEALTH2_POS]
        opponent_health_bgr = frame[HEALTH1_POS]

        p_damage, p_died = self.player_healthtracker.update(player_health_bgr)
        o_damage, o_died = self.opponent_healthtracker.update(opponent_health_bgr)

        current_time = time.time()
        # if death occurred, start the reward override timer
        if p_died or o_died:
            self.last_death_time = current_time

        # check if we're still within the reward override period
        if current_time - self.last_death_time < 5:
            return 1.0
        

        
        return 1 - action_idx/NUM_ACTIONS  # placeholder

    def check_done(self):
        # Detect if game is over
        return False

class HealthTracker:
    def __init__(self, dmg_color_lut):

        self.damage = 0
        self.dmg_color_lut = dmg_color_lut

    def update(self, current_color):
        """
        Compute the magnitude of health color change (~ damage taken) and check if player died (color reset to white).
        
        :param current_color: tuple/list BGR color from health bar
        :return: (magnitude: float, died: bool)
        """
        died = False
        current_color = np.array(current_color, dtype=float)

        new_damage = self.get_damage_from_color(current_color)

        # if color is not in dmg_lut -> get_damage_from_color returns -1
        if new_damage == -1:
            damage_diff = 0
            new_damage = 0
            # if unkown color is some type of red the player died
            died = self.is_red(current_color)

        else:
            damage_diff = new_damage - self.damage

            # if the damage_diff is negative the player also died
            if damage_diff < 0:
                died = True
                damage_diff = 0
                new_damage = 0

        self.damage = new_damage
  
        return damage_diff, died
    
    def get_damage_from_color(self, color, tolerance=3):
        """
        Given a color and a list of known health colors, return the closest matching damage value.
        """
        color = np.array(color, dtype=int)
        color_array = np.array(self.dmg_color_lut, dtype=int)

        # Calculate Euclidean distances
        distances = np.linalg.norm(color_array - color, axis=1)

        # Get the closest match within tolerance
        min_index = np.argmin(distances)
        if distances[min_index] <= tolerance:
            return min_index
        else:
            return -1
        
    def is_red(self, color):
        """
        Detects red in HSV space.
        Accepts BGR tuple and returns True if it's red.
        """
        hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv

        # Red is around 0-10 and 160-180 in hue
        return (h <= 10 or h >= 160) and s >= 50 and v >= 50