import pickle
import cv2
import numpy as np
import os
from BrawlhallaController import BrawlhallaController

class DataViewer:
    """
    A class for visualizing expert demonstration episodes for Brawlhalla RL agent training.
    Loads and displays the last 2 collected expert episodes, allowing frame-by-frame navigation.
    """
    def __init__(self, data_dir='data/', font_scale=1.0, font_thickness=2, text_padding=15, line_spacing=25):
        """
        Initialize the DataViewer, load episodes, and set up the OpenCV window and trackbar.
        """
        self.data_dir = data_dir
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_padding = text_padding
        self.line_spacing = line_spacing

        self.expert_episodes = self._load_last_episodes(num_episodes=2)
        if not self.expert_episodes:
            print("No expert episodes found.")
            self.frame_map = []
            return

        self.controller = BrawlhallaController()
        self._build_frame_map()

        self.current_idx = 0
        self.max_idx = self.total_frames - 1
        self.num_episodes = len(self.expert_episodes)

        cv2.namedWindow("Data Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Data Viewer", 1280, 720)
        cv2.createTrackbar('Frame', 'Data Viewer', 0, self.max_idx, self._on_trackbar_change)
        self._display_frame()

    def _load_last_episodes(self, num_episodes=2):
        """
        Load the last `num_episodes` expert episodes from the data directory.
        Returns a list of loaded episode dicts.
        """
        episode_files = []
        for filename in os.listdir(self.data_dir):
            if filename.startswith('expert_episode_') and filename.endswith('.pkl'):
                try:
                    episode_num = int(filename.split('_')[2].split('.')[0])
                    episode_files.append((episode_num, os.path.join(self.data_dir, filename)))
                except ValueError:
                    print(f"Warning: Skipping malformed episode file: {filename}")
        episode_files.sort()
        episode_files = episode_files[-num_episodes:]

        print("Loaded episodes:")
        for _, filepath in episode_files:
            print(f"  {os.path.basename(filepath)}")

        episodes = []
        for _, filepath in episode_files:
            try:
                with open(filepath, 'rb') as f:
                    episode_data = pickle.load(f)
                    episodes.append(episode_data)
            except Exception as e:
                print(f"Error loading episode from {filepath}: {e}")
        return episodes

    def _build_frame_map(self):
        """
        Build a mapping from global frame index to (episode_idx, timestep_idx).
        Also computes total_frames.
        """
        self.frame_map = []
        self.total_frames = 0
        for episode_idx, episode in enumerate(self.expert_episodes):
            num_timesteps = episode["obs"].shape[0]
            for timestep_idx in range(num_timesteps):
                self.frame_map.append((episode_idx, timestep_idx))
            self.total_frames += num_timesteps

    def _on_trackbar_change(self, val):
        """
        Callback for OpenCV trackbar to update the displayed frame.
        """
        self.current_idx = val
        self._display_frame()

    def _display_frame(self):
        """
        Display the current frame with overlayed episode/timestep and action info.
        """
        if not self.frame_map:
            return
        episode_idx, timestep_idx = self.frame_map[self.current_idx]
        episode = self.expert_episodes[episode_idx]
        img = episode["obs"][timestep_idx]
        current_action_idx = episode["actions"][timestep_idx]
        num_timesteps_in_episode = episode["obs"].shape[0]

        # Convert image to displayable format (H, W, C) for color, or (H, W) for grayscale
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            if img.shape[0] == 1:
                display_img = np.squeeze(img, axis=0).copy().astype(np.uint8)
            else:
                display_img = np.transpose(img, (1, 2, 0)).copy().astype(np.uint8)
        elif img.ndim == 2:
            display_img = img.copy().astype(np.uint8)
        else:
            print(f"Warning: Unexpected image shape {img.shape} at index {self.current_idx}")
            return

        display_img = cv2.resize(display_img, (1920, 1080))
        current_action_name = self.controller.ACTION_LUT[current_action_idx]

        text_lines = [
            f"Episode: {episode_idx+1}/{self.num_episodes} | Timestep: {timestep_idx+1}/{num_timesteps_in_episode}",
            f"Current Action: {current_action_name}"
        ]

        img_height = display_img.shape[0]
        y_start = img_height - self.text_padding
        for i, line in enumerate(reversed(text_lines)):
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
            y = y_start - (i * (text_height + self.line_spacing))
            cv2.rectangle(display_img, (10, y - text_height - baseline), (10 + text_width, y + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(display_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

        cv2.imshow("Data Viewer", display_img)

    def run(self):
        """
        Start the OpenCV event loop for the data viewer.
        """
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = DataViewer()
    viewer.run() 