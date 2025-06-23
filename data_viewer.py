import pickle
import cv2
import numpy as np
from BrawlhallaController import BrawlhallaController
from config import config # Import config
config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"] = 512, 288
import os

class DataViewer:
    def __init__(self, data_dir='data/',
                 font_scale=1.0,
                 font_thickness=2,
                 text_padding=15,
                 line_spacing=25):

        self.expert_episodes = self._load_all_episodes(data_dir)

        if not self.expert_episodes:
            print("No expert episodes found.")
            return

        self.controller = BrawlhallaController()

        self.total_frames = 0
        self.frame_map = [] # Maps global_idx -> (episode_idx, timestep_idx)
        for episode_idx, episode in enumerate(self.expert_episodes):
            num_timesteps = episode["obs"]["image"].shape[0]
            for timestep_idx in range(num_timesteps):
                self.frame_map.append((episode_idx, timestep_idx))
            self.total_frames += num_timesteps

        self.num_episodes = len(self.expert_episodes)
        
        self.current_idx = 0
        self.max_idx = self.total_frames - 1

        # Configurable text display parameters
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_padding = text_padding
        self.line_spacing = line_spacing

        cv2.namedWindow("Data Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Data Viewer", 1280, 720)

        cv2.createTrackbar('Frame', 'Data Viewer', 0, self.max_idx, self._on_trackbar_change)
        self._display_frame()

    def _load_all_episodes(self, data_dir, max_frames):
        episodes = []
        episode_files = []
        current_loaded_frames = 0

        # Collect all expert_episode_X.pkl files
        for filename in os.listdir(data_dir):
            if filename.startswith('expert_episode_') and filename.endswith('.pkl'):
                # Extract the numeric part for sorting
                try:
                    episode_num = int(filename.split('_')[2].split('.')[0])
                    episode_files.append((episode_num, os.path.join(data_dir, filename)))
                except ValueError:
                    print(f"Warning: Skipping malformed episode file: {filename}")
                    continue
        
        # Sort files numerically by episode number
        episode_files.sort()

        for episode_num, filepath in episode_files:
            try:
                with open(filepath, 'rb') as f:
                    episode_data = pickle.load(f)
                    num_timesteps = episode_data["obs"]["image"].shape[0]

                    if current_loaded_frames + num_timesteps <= max_frames:
                        episodes.append(episode_data)
                        current_loaded_frames += num_timesteps
                    else:
                        # If adding the full episode exceeds max_frames, truncate it
                        remaining_frames = max_frames - current_loaded_frames
                        if remaining_frames > 0:
                            truncated_episode_data = {
                                "obs": {
                                    "image": episode_data["obs"]["image"][:remaining_frames],
                                    "last_executed_action": episode_data["obs"]["last_executed_action"][:remaining_frames]
                                },
                                "actions": episode_data["actions"][:remaining_frames],
                                "dones": episode_data["dones"][:remaining_frames]
                            }
                            episodes.append(truncated_episode_data)
                            current_loaded_frames += remaining_frames
                        print(f"Reached max_frames limit ({max_frames}). Stopping further episode loading.")
                        break # Stop loading further episodes
            except Exception as e:
                print(f"Error loading episode from {filepath}: {e}")
                continue
        return episodes

    def _on_trackbar_change(self, val):
        self.current_idx = val
        self._display_frame()

    def _display_frame(self):
        if self.expert_episodes is None or not self.frame_map: # Check if data was loaded successfully
            return

        episode_idx, timestep_idx = self.frame_map[self.current_idx]

        episode = self.expert_episodes[episode_idx]
        img = episode["obs"]["image"][timestep_idx]
        current_action_idx = episode["actions"][timestep_idx]
        num_timesteps_in_episode = episode["obs"]["image"].shape[0] # Get timesteps for current episode
        
        # Convert image to displayable format (H, W, C) for color, or (H, W) for grayscale
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # Check if it's (C, H, W)
            if img.shape[0] == 1:
                display_img = np.squeeze(img, axis=0).copy().astype(np.uint8) # Grayscale
            else:
                display_img = np.transpose(img, (1, 2, 0)).copy().astype(np.uint8) # Color
        elif img.ndim == 2: # Already (H, W) grayscale
            display_img = img.copy().astype(np.uint8)
        else:
            print(f"Warning: Unexpected image shape {img.shape} at index {self.current_idx}")
            return

        display_img = cv2.resize(display_img, (1920, 1080)) # Resize using config

        # Get the current action name
        current_action_name = self.controller.ACTION_LUT[current_action_idx][0]
        
        # Prepare text overlay
        text_lines = []
        text_lines.append(f"Episode: {episode_idx+1}/{self.num_episodes} | Timestep: {timestep_idx+1}/{num_timesteps_in_episode}")
        text_lines.append(f"Current Action: {current_action_name}")

        # Add text overlay to image (bottom-left)
        img_height = display_img.shape[0]
        
        # Calculate starting y position for the bottom-most line of text
        y_start = img_height - self.text_padding

        # Iterate through text lines in reverse order to draw from bottom up
        for i, line in enumerate(reversed(text_lines)):
            # Get text size for current line to calculate dynamic line spacing
            (text_width, text_height), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
            
            # Dynamically calculate line position based on text height and custom line_spacing attribute
            # The `baseline` is added to ensure text is fully visible above the rectangle
            y = y_start - (i * (text_height + self.line_spacing)) 
            
            # Draw black rectangle as backdrop
            cv2.rectangle(display_img, (10, y - text_height - baseline), (10 + text_width, y + baseline), (0, 0, 0), cv2.FILLED)
            
            cv2.putText(display_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

        cv2.imshow("Data Viewer", display_img)

    def run(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = DataViewer()
    viewer.run() 