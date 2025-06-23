import pickle
import cv2
import numpy as np
import os
import threading
import time
from copy import deepcopy
from pynput import keyboard
from pynput.keyboard import Key, KeyCode, Listener

from BrawlhallaEnv import BrawlhallaEnv
from config import config


class ExpertDataManager:
    """
    Collects and saves expert demonstration data for Brawlhalla RL agent training.
    Handles keyboard input, environment stepping, and episode management.
    """
    def __init__(self, config):
        """
        Initialize the data manager, environment, and keyboard listener.
        """
        self.config = config
        self.episode_counter = self._get_next_episode_number()
        self.env = BrawlhallaEnv(config=config)
        self.env.observe_only = True
        self.obs, self.info = self.env.reset()

        self.current_episode_images = []
        self.current_episode_actions = []
        self.current_episode_dones = []
        self.saving_threads = []

        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.key_history = {"pressed": set(), "released": set()}

        self.valid_actions, self.action_idx_map = self._build_action_maps()

    def _normalize_key(self, key):
        if isinstance(key, KeyCode) and key.char:
            return KeyCode.from_char(key.char.lower())
        return key

    def _on_press(self, key):
        self.key_history["pressed"].add(self._normalize_key(key))

    def _on_release(self, key):
        self.key_history["released"].add(self._normalize_key(key))

    def _to_canonical_key(self, key):
        if isinstance(key, keyboard.Key):
            return key
        elif isinstance(key, keyboard.KeyCode):
            return key.char.lower() if key.char else key
        elif isinstance(key, str):
            return key.lower()
        return key

    def _get_next_episode_number(self):
        """
        Determines the next available episode number based on existing files.
        """
        max_episode_num = -1
        for filename in os.listdir('data/'):
            if filename.startswith('expert_episode_') and filename.endswith('.pkl'):
                try:
                    episode_num = int(filename.split('_')[2].split('.')[0])
                    if episode_num > max_episode_num:
                        max_episode_num = episode_num
                except ValueError:
                    continue
        return max_episode_num + 1

    def _build_action_maps(self):
        """
        Build valid actions dict: action_idx -> set of canonical keys.
        """
        valid_actions = []
        action_idx_map = {}
        for action_idx, mapped_keys_list in self.env.controller.idx_to_keys.items():
            canonical_mapped_keys = frozenset(self._to_canonical_key(k) for k in mapped_keys_list)
            valid_actions.append(canonical_mapped_keys)
            action_idx_map[canonical_mapped_keys] = action_idx
        return valid_actions, action_idx_map

    def get_best_valid_action(self, active_keys_from_controller):
        """
        Maps a set of currently active pynput keys to a single action index from ACTION_LUT.
        Actions are prioritized by their order in the ACTION_LUT.
        """
        if active_keys_from_controller in (set([KeyCode.from_char('j')]), set([KeyCode.from_char('k')])):
            active_keys_from_controller.add(KeyCode.from_char('w'))
        canonical_active_keys = frozenset(self._to_canonical_key(k) for k in active_keys_from_controller)
        matched_sets = [s for s in self.valid_actions if s.issubset(canonical_active_keys)]
        if matched_sets:
            best_set = matched_sets[:2][-1]
            return self.action_idx_map[best_set]
        return 0  # Fallback to 'idle'

    def save_example_img(self):
        """
        Save an example raw observation frame as a PNG file.
        """
        if self.obs is not None:
            image_to_save = self.obs
            if self.env.img_shape[0] == 1:
                image_to_save = np.squeeze(image_to_save, axis=0)
            else:
                image_to_save = np.transpose(image_to_save, (1, 2, 0))
            filename = "example_obs_.png"
            cv2.imwrite(filename, image_to_save)
            print("Saved", filename)
        else:
            print("Warning: Could not capture an example frame to save.")

    def _perform_save(self, episode_data, filename):
        """
        Actual file saving logic, run in a separate thread.
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(episode_data, f)
            print(f"Saved episode {self.episode_counter} to {filename} with {len(episode_data['obs'])} timesteps.")
        except Exception as e:
            print(f"Error saving episode to {filename}: {e}")

    def _save_current_episode(self):
        """
        Save the current episode's data to disk and reset buffers.
        """
        if not self.current_episode_images:
            return
        episode_data = {
            "obs": np.stack(self.current_episode_images, axis=0),
            "actions": np.array(self.current_episode_actions, dtype=np.int64),
            "dones": np.array(self.current_episode_dones, dtype=bool),
        }
        filename = os.path.join('data', f'expert_episode_{self.episode_counter}.pkl')
        save_thread = threading.Thread(target=self._perform_save, args=(episode_data, filename))
        save_thread.start()
        self.saving_threads.append(save_thread)
        self.episode_counter += 1
        self.current_episode_images = []
        self.current_episode_actions = []
        self.current_episode_dones = []

    def collect_data(self):
        """
        Main loop for collecting expert data. Steps the environment, records actions, and saves episodes.
        """
        print("Starting Data Collection")
        step_count = 0
        last_time = time.time()
        fps_history = []
        try:
            while True:
                obs, reward, done, truncated, info = self.env.step(0)

                # Get all pressed keys and remove released keys from history
                _key_history = deepcopy(self.key_history)
                forced_released_keys = {Key.shift, Key.space}
                _key_history["released"].update(forced_released_keys)
                self.key_history["pressed"].difference_update(_key_history["released"])
                self.key_history["released"].clear()

                actions = _key_history["pressed"]
                action_idx = self.get_best_valid_action(actions)

                self.current_episode_images.append(obs)
                self.current_episode_actions.append(action_idx)
                self.current_episode_dones.append(done)
                step_count += 1
                now = time.time()
                if now - last_time >= 1.0:
                    fps = step_count / (now - last_time)
                    fps_history.append(fps)
                    step_count = 0
                    last_time = now
                if done:
                    self._save_current_episode()
                    self.obs, self.info = self.env.reset()

        except KeyboardInterrupt:
            print("Stopping data collection...")
        finally:
            if self.current_episode_images:
                self._save_current_episode()
            print("Waiting for background saving to complete...")
            for thread in self.saving_threads:
                thread.join()
            print("All background saving threads completed.")
            if fps_history:
                avg_fps = sum(fps_history) / len(fps_history)
                std_fps = (sum((x - avg_fps) ** 2 for x in fps_history) / len(fps_history)) ** 0.5
                print(f"Average FPS: {avg_fps:.2f}, Std: {std_fps:.2f}")
            self.env.close()
            print("Environment closed and controller listener stopped.")

if __name__ == "__main__":
    manager = ExpertDataManager(config)
    manager.collect_data()