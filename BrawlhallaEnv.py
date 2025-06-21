import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import atexit

from config import *
from ScreenRecorder import ScreenRecorder
from BrawlhallaController import BrawlhallaController

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
    def __init__(self, img_shape=(3,1080,1920)):
        super().__init__()

        self.controller = BrawlhallaController()
        self.img_shape = img_shape
        self.num_actions = len(self.controller.ACTION_LUT)

        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, img_shape, dtype=np.uint8),
            'last_executed_action': spaces.Box(low=0, high=self.num_actions - 1, shape=(1,), dtype=np.int64) # use 1d box instead of discrete space to prevent one-hot encoding
        })

        self.action_space = spaces.Discrete(self.num_actions)

        self.recorder = ScreenRecorder(fps=REC_FPS, region=None, buffer_size=8, render=False, monitor_id=GAME_MONITOR_ID, resolution=(img_shape[2], img_shape[1]))
        self.recorder.start()
        time.sleep(.1)

        # Save an example image
        example_frame = self.recorder.get_latest_frame()
        if example_frame is not None:
            resized_example_frame = cv2.resize(example_frame, (img_shape[2], img_shape[1]))
            if img_shape[0] == 1:
                processed_frame = cv2.cvtColor(resized_example_frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("example_image.png", processed_frame)
            else:
                cv2.imwrite("example_image.png", resized_example_frame)
        else:
            print("Warning: Could not capture an example frame to save.")

        atexit.register(self.close)

        self.step_time = None

        self.player_healthtracker = HealthTracker(dmg_color_lut=DMG_COLOR_LUT)
        self.opponent_healthtracker = HealthTracker(dmg_color_lut=DMG_COLOR_LUT)

        self.last_death_time = 0

    def close(self):
        if self.recorder is not None:
            self.recorder.stop()
            self.controller.release_all_keys()
            cv2.destroyAllWindows()

    def reset(self, seed=0):
        # Reset game or restart episode
        time.sleep(4) # wait for respawn
        self.controller.release_all_keys()
        print(self.controller.pressed_keys)
        self.controller.reset_o_health()
        time.sleep(0.5) 

        obs = self._get_obs(action_idx=5) # Pass default action for reset (5 = idle)
        info = {}
        return obs, info

    def step(self, action_idx):
       
        self.perform_action(action_idx)

        # Enforce LEARNING_FPS to keep a consistent fps during training since we dont have full control over the environment
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
        obs = self._get_obs(action_idx)

        reward = self._compute_reward(action_idx)
        terminated = (reward <= -0.8)
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self, action_idx):

        
    
        frame = self.recorder.get_latest_frame()
        resized_frame = cv2.resize(frame, (self.img_shape[2], self.img_shape[1]))

        if self.img_shape[0] == 1: 
            image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)[np.newaxis, :, :]
        else:
            transposed_frame = np.transpose(resized_frame, (2, 0, 1))
            image = transposed_frame

        if image.shape != self.img_shape:
            raise ValueError(
                f"Image shape mismatch. Expected {self.img_shape}, but got {image.shape}"
            )

        return {"image": image, "last_executed_action": np.array([action_idx], dtype=np.int64)}

    def perform_action(self, action_idx):
        keys_to_press = set(self.controller.ACTION_MAPPER.get(action_idx, []))

        # Determine which keys to release (keys that were pressed but are not in the current action)
        keys_to_release = self.controller.pressed_keys - keys_to_press
        self.controller.release_keys(keys_to_release)

        self.controller.press_keys(keys_to_press)

    def _compute_reward(self, action_idx):
        frame = self.recorder.get_latest_frame()
        
        # get colors at the position of the respective healthbars
        player_health_bgr = frame[HEALTH2_POS]
        opponent_health_bgr = frame[HEALTH1_POS]

        p_damage, p_died = self.player_healthtracker.update(player_health_bgr)
        o_damage, o_died = self.opponent_healthtracker.update(opponent_health_bgr)

        current_time = time.time()

        # check if we're still within the respawn period
        if current_time - self.last_death_time < 4:
            return 0.0
        
        # if death occurred, start the reward override timer for respawning
        if p_died or o_died:
            self.last_death_time = current_time
        
        r = 0

        if o_died:
            r += 1
        
        if p_died:
            r -= 1
        
        r += o_damage / 50 - p_damage / 50

        attacking_actions = set([7,8,12,13,16,17,21,22]) # all action_idx that are attacks (see ACTION_LUT in config)
        movement_actions = set([0,1,2,3,4,10,11,19,20]) # all action_idx that are movement only
        if action_idx in movement_actions:
            r += 0.04
            
        elif action_idx in attacking_actions and o_damage == 0: # if attack missed
            r -= 0.04

        return r

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
        return (h <= 10 or h >= 160) and s >= 10 and v >= 100-0.5*s-25