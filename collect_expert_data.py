import pickle
import time
from BrawlhallaEnv import BrawlhallaEnv
from pynput import keyboard
import cv2  # Import opencv-python
import numpy as np # Import numpy

# Storage for (timestamp, obs, action) tuples
observation_data = []
# Storage for (timestamp, event_type, key) tuples
key_event_log = []

# Reverse mapping from key to action name (based on controller.py SPECIAL_KEYS)
key_to_action = {
    'a': 'left',
    'd': 'right',
    'w': 'up',
    's': 'down',
    'Key.space': 'jump',
    'j': 'light',
    'k': 'heavy',
    'u': 'throw',
    'Key.shift': 'dodge', 
}

down_keys = set()

def key_to_log_str(key):
    # Normalize to lowercase for alphabetic keys
    try:
        return key.char.lower()
    except AttributeError:
        return str(key)

def on_press(key):
    norm_key = key_to_log_str(key)
    if norm_key not in down_keys and norm_key in list(key_to_action.keys()):
        print("key press:", key_to_action[norm_key])
        key_event_log.append((time.time(), 'press', norm_key))
        down_keys.add(norm_key)

def on_release(key):
    norm_key = key_to_log_str(key)
    if norm_key in down_keys and norm_key in list(key_to_action.keys()):
        print("key release:", key_to_action[norm_key])
        key_event_log.append((time.time(), 'release', norm_key))
        down_keys.remove(norm_key)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

env = BrawlhallaEnv(img_shape=(3,288,512))
obs, info = env.reset()

# Save an example raw observation frame
if obs is not None and "image" in obs:
    image_to_save = obs["image"]
    if env.img_shape(0) == 1:
        # Grayscale image, remove channel dimension (C, H, W) -> (H, W)
        image_to_save = np.squeeze(image_to_save, axis=0)
    else:
        # Color image, transpose from (C, H, W) to (H, W, C) for OpenCV
        image_to_save = np.transpose(image_to_save, (1, 2, 0))

    cv2.imwrite("example_image.png", image_to_save)
    print("Saved example_image.png")
else:
    print("Warning: Could not capture an example frame to save.")

try:
    while True:
        
        # Store observation with timestamp
        obs, reward, done, truncated, info = env.step(5)
        observation_data.append((time.time(), obs["image"]))

        if done:
            obs, info = env.reset()
except KeyboardInterrupt:
    print("Stopping data collection...")

with open('data/expert_data_obs.pkl', 'wb') as f:
    pickle.dump(observation_data, f)
with open('data/expert_data_keys.pkl', 'wb') as f:
    pickle.dump(key_event_log, f)
print(f"Saved {len(observation_data)} observations and {len(key_event_log)} key events.")