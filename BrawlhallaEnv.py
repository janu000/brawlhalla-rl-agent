import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import atexit

from ScreenRecorder import ScreenRecorder
from BrawlhallaController import BrawlhallaController

DMG_COLOR_LUT = []
with open("utils/healthcolors.txt", 'r') as f:
    for line in f:
        # Remove brackets and split into integers
        line = line.strip().replace('[', '').replace(']', '')
        if line:
            parts = list(map(int, line.split()))
            if len(parts) == 3:
                DMG_COLOR_LUT.append(tuple(parts))
                
class BrawlhallaEnv(gym.Env):
    def __init__(self, config: dict):
        super().__init__()

        self.controller = BrawlhallaController()
        self.img_shape = config["IMAGE_CHANNELS"], config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"]
        self.num_actions = len(self.controller.ACTION_LUT)
        self.max_eps_len = config["MAX_EPS_LEN"]

        self.observation_space = spaces.Box(0, 255, self.img_shape, dtype=np.uint8)

        self.action_space = spaces.Discrete(self.num_actions)
        
        self.step_duration = 1 / config["STEP_FPS"]
        self.observe_only = False

        self.p_health_pos = config["HEALTH_POS1"]
        self.o_health_pos = config["HEALTH_POS2"]

        self.player_healthtracker = HealthTracker(dmg_color_lut=DMG_COLOR_LUT)
        self.opponent_healthtracker = HealthTracker(dmg_color_lut=DMG_COLOR_LUT)

        atexit.register(self.close)

        self.step_counter = 0
        self.step_timer = 0
        self.recorder = ScreenRecorder(
            fps=config["REC_FPS"],
            region=None,
            buffer_size=8,
            render=False,
            monitor_id=config["GAME_MONITOR_ID"],
            resolution=(self.img_shape[2], self.img_shape[1]),
            grayscale=(config["IMAGE_CHANNELS"] == 1))
        
        self.recorder.start()

    def close(self):
        if self.recorder is not None:
            self.recorder.stop()
            self.controller.release_all_keys()
            cv2.destroyAllWindows()

    def reset(self, seed=0):
        print("reset")
        # Reset opponent health and wait for player respawn
        self.controller.release_all_keys(unjam=True)
        # right_key = self.controller.name_to_keys["right"]
        # self.controller.press_keys(right_key)
        time.sleep(2)
        # self.controller.release_keys(right_key)
        self.controller.reset_o_health()
        time.sleep(3)
        print("continue")

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action_idx):
        self.step_counter += 1

        if not self.observe_only:
            self.perform_action(action_idx)

        # Enforce STEP_FPS to keep a consistent step fps since we dont have full control over the continous environment
        elapsed_time = time.monotonic() - self.step_timer
        sleep_duration = self.step_duration - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.step_timer = time.monotonic()

        # Read next observation
        obs = self._get_obs()

        reward = self._compute_reward(action_idx)
        terminated = (reward <= -0.8) or (self.step_counter >= self.max_eps_len)
        if terminated:
            self.step_counter = 0
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):

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
        return image

    def perform_action(self, action_idx):
        self.controller.execute_action(action_idx)

    def _compute_reward(self, action_idx):
        frame = self.recorder.get_latest_frame()
        
        # get colors at the position of the respective healthbars
        player_health_bgr = frame[self.p_health_pos] # takes in position as (y,x)
        opponent_health_bgr = frame[self.o_health_pos] # take in position as (y,x)

        p_damage, p_died = self.player_healthtracker.update(player_health_bgr)
        o_damage, o_died = self.opponent_healthtracker.update(opponent_health_bgr)
        
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