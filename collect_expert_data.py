import pickle
from pynput import keyboard
import cv2  # Import opencv-python
import numpy as np # Import numpy
from pynput.keyboard import Key, KeyCode, Listener
from copy import deepcopy
from itertools import combinations
import time
import os # Import os for path joining
import threading # Import threading

from BrawlhallaEnv import BrawlhallaEnv


from config import config
config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"] = 256, 144


class ExpertDataManager:
    def __init__(self, config):
        # List of lists, where each inner list represents an episode
        self.config = config

        self.episode_images = []
        self.episode_actions = []
        self.episode_dones = []

        # Current episode data being collected
        self.current_episode_images = []
        self.current_episode_actions = []
        self.current_episode_dones = []
        self.saving_threads = [] # List to hold saving threads

        # Determine the starting episode number
        self.episode_counter = self._get_next_episode_number()

        # Initialize BrawlhallaEnv, which internally creates BrawlhallaController
        self.env = BrawlhallaEnv(config=config)
        self.env.observe_only = True
        self.obs, self.info = self.env.reset()

        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.key_history = {"pressed": set(), "released": set()}

        # Build valid actions dict: action_idx -> set of canonical keys
        self.valid_actions = []
        self.action_idx_map = {}
        for action_idx, mapped_keys_list in self.env.controller.idx_to_keys.items():
            canonical_mapped_keys = frozenset(self._to_canonical_key(k) for k in mapped_keys_list)
            self.valid_actions.append(canonical_mapped_keys)
            self.action_idx_map[canonical_mapped_keys] = action_idx

    def _normalize_key(self, key):
        if isinstance(key, KeyCode) and key.char:
            return KeyCode.from_char(key.char.lower())
        return key  # Leave Key (e.g., Key.space) unchanged

    def _on_press(self, key):
        self.key_history["pressed"].add(self._normalize_key(key))

    def _on_release(self, key):
        self.key_history["released"].add(self._normalize_key(key))

    def _to_canonical_key(self, key):
        """Converts a pynput key object or string to a canonical form for comparison."""
        if isinstance(key, keyboard.Key):
            return key # e.g., Key.space, Key.shift_l
        elif isinstance(key, keyboard.KeyCode):
            # Convert KeyCode to its character or itself if no char (e.g., F1)
            return key.char.lower() if key.char else key
        elif isinstance(key, str):
            return key.lower() # Already a char string
        return key # Fallback for unexpected types

    def _get_next_episode_number(self):
        """Determines the next available episode number based on existing files."""
        max_episode_num = -1
        for filename in os.listdir('data/'):
            if filename.startswith('expert_episode_') and filename.endswith('.pkl'):
                try:
                    episode_num = int(filename.split('_')[2].split('.')[0])
                    if episode_num > max_episode_num:
                        max_episode_num = episode_num
                except ValueError:
                    # Ignore files with malformed names
                    continue
        return max_episode_num + 1

    def get_best_valid_action(self, active_keys_from_controller):
        """
        Maps a set of currently active pynput keys (from BrawlhallaController)
        to a single action index from ACTION_LUT.
        Actions are prioritized by their order in the ACTION_LUT
        """
        # Override light or heavy attack only to attack + w
        if active_keys_from_controller in (set([KeyCode.from_char('j')]), set([KeyCode.from_char('k')])):
            active_keys_from_controller.add(KeyCode.from_char('w'))

        canonical_active_keys = frozenset(self._to_canonical_key(k) for k in active_keys_from_controller)
        
        matched_sets = [
            s for s in self.valid_actions
            if s.issubset(canonical_active_keys)
        ]

        # Return the one with highest priority (lowest index in list)
        if matched_sets:
            best_set = matched_sets[:2][-1]  # valid_actions is ordered except idle at idx 0
            return self.action_idx_map[best_set]

        # Fallback to 'idle' 
        return 0


    def save_example_img(self):
        # Save an example raw observation frame
        if self.obs is not None and "image" in self.obs:
            image_to_save = self.obs["image"]
            if self.env.img_shape[0] == 1:
                # Grayscale image, remove channel dimension (C, H, W) -> (H, W)
                image_to_save = np.squeeze(image_to_save, axis=0)
            else:
                # Color image, transpose from (C, H, W) to (H, W, C) for OpenCV
                image_to_save = np.transpose(image_to_save, (1, 2, 0))

            # This action is from the environment's last reported action, not derived from active keys.
            # This is fine for example image filename.
            if "last_executed_action" in self.obs:
                action = self.env.controller.ACTION_LUT[self.obs["last_executed_action"].item()][0]
                filename = "example_obs_" + action + ".png"
            cv2.imwrite(filename, image_to_save)
            print("Saved ", filename)
        else:
            print("Warning: Could not capture an example frame to save.")

    def _perform_save(self, episode_data, filename):
        """Actual file saving logic, run in a separate thread."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(episode_data, f)
            print(f"Saved episode {self.episode_counter} to {filename} with {len(episode_data['obs']['image'])} timesteps.")
        except Exception as e:
            print(f"Error saving episode to {filename}: {e}")

    def _save_current_episode(self):
        if not self.current_episode_images:
            return # No data to save

        episode_data = {
            "obs": {
                "image": np.stack(self.current_episode_images, axis=0), # T, C, H, W
                "last_executed_action": np.expand_dims(np.array(self.current_episode_actions, dtype=np.int64), axis=-1) # T, 1
            },
            "actions": np.array(self.current_episode_actions, dtype=np.int64), # T
            "dones": np.array(self.current_episode_dones, dtype=bool), # T
        }

        filename = os.path.join('data', f'expert_episode_{self.episode_counter}.pkl')
        
        # Create and start a new thread to save the episode data
        save_thread = threading.Thread(target=self._perform_save, args=(episode_data, filename))
        save_thread.start()
        self.saving_threads.append(save_thread)

        self.episode_counter += 1
        # Reset for next episode
        self.current_episode_images = []
        self.current_episode_actions = []
        self.current_episode_dones = []

    def collect_data(self):
        print("Recording started.")
        try:
            while True:
                # Step the environment to get the next observation. 
                # The '5' is a dummy action_idx as self.env.observe_only = True means env doesn't execute it.
                obs, reward, done, truncated, info = self.env.step(5) 
                _key_history = deepcopy(self.key_history)

                # Force specific keys to always be released
                forced_released_keys = {Key.shift, Key.space}
                _key_history["released"].update(forced_released_keys)

                # Clear released keys from pressed set and clear released history
                self.key_history["pressed"].difference_update(_key_history["released"])
                self.key_history["released"].clear()


                actions = _key_history["pressed"]
                # Determine the corresponding action index from ACTION_LUT
                action_idx = self.get_best_valid_action(actions)
                print(self.env.controller.ACTION_LUT[action_idx])

                self.current_episode_images.append(obs["image"])
                self.current_episode_actions.append(action_idx)
                self.current_episode_dones.append(done)

                if done:
                    self._save_current_episode()
                    self.obs, self.info = self.env.reset()
        except KeyboardInterrupt:
            print("Stopping data collection...")
        finally:
            if self.current_episode_images:
                self._save_current_episode()
            
            # Wait for all saving threads to complete
            print("Waiting for background saving to complete...")
            for thread in self.saving_threads:
                thread.join()
            print("All background saving threads completed.")

            self.env.close()
            print("Environment closed and controller listener stopped.")

if __name__ == "__main__":
    manager = ExpertDataManager(config)
    manager.collect_data()