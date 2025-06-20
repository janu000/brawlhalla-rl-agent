import mss
import cv2
import numpy as np
import os
import datetime
import threading
from pynput import keyboard
import time # Import time

# Configuration
MONITOR_NUMBER = 2  # Change this to your second monitor's number
OUTPUT_DIR = "data"
FPS = 24.0 # Frames per second for the video recording
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for controlling recording
is_recording = False # Initial state: not recording
keyboard_events = []
current_video_writer = None # This will hold the cv2.VideoWriter object
screen_thread_instance = None # To hold the screen recording thread instance

recording_state_lock = threading.Lock() # To protect shared recording state variables

def save_keyboard_events():
    """Saves the accumulated keyboard events to a file."""
    global keyboard_events
    if keyboard_events:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        keyboard_filename = os.path.join(OUTPUT_DIR, f"keyboard_inputs_{timestamp}.txt")
        with open(keyboard_filename, "w") as f:
            for event in keyboard_events:
                f.write(event + "\n")
        print(f"Keyboard inputs saved to {keyboard_filename}")
        keyboard_events = [] # Clear for next session

def record_screen_loop():
    """Continuously records the screen while is_recording is True."""
    global is_recording, current_video_writer

    sct = mss.mss() # Initialize mss.mss() inside the thread

    monitor = sct.monitors[MONITOR_NUMBER]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Prepare for the current recording session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(OUTPUT_DIR, f"screen_record_{timestamp}.mp4")
    
    with recording_state_lock:
        current_video_writer = cv2.VideoWriter(video_filename, fourcc, FPS, (TARGET_WIDTH, TARGET_HEIGHT))
        print(f"Starting screen recording to {video_filename}.")

    # Precise FPS control
    time_per_frame = 1 / FPS
    start_time = time.time() # Reset start time for next frame

    while is_recording: # Loop while the global flag is True
        try:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # Corrected: mss on Windows outputs BGRA, convert to BGR

            resized_img = cv2.resize(img_bgr, (TARGET_WIDTH, TARGET_HEIGHT))
            
            with recording_state_lock: # Ensure we don't try to write if writer is released
                if current_video_writer:
                    current_video_writer.write(resized_img)
                else:
                    break # Writer was released by another thread
        except Exception as e:
            print(f"Error during screen recording: {e}")
            with recording_state_lock:
                is_recording = False # Stop recording on error
            break

        # Precise FPS control
        end_time = time.time()
        time_taken = end_time - start_time # Time taken for one frame processing
        if time_taken < time_per_frame:
            time.sleep(time_per_frame - time_taken)
        start_time = time.time() # Reset start time for next frame

    # After the loop, the recording has stopped
    with recording_state_lock:
        if current_video_writer:
            current_video_writer.release()
            current_video_writer = None
    print("Screen recording session stopped.") # Moved outside the lock for clearer output.
    save_keyboard_events() # Save events here, after the screen recording has definitely stopped and released resources.


def on_press(key):
    global keyboard_events
    if is_recording: # Only record keys if currently recording
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        try:
            keyboard_events.append(f"{timestamp}: Key {key.char} pressed")
        except AttributeError:
            keyboard_events.append(f"{timestamp}: Special key {key} pressed")

def on_release(key):
    global is_recording, screen_thread_instance

    if key == keyboard.Key.f11:
        with recording_state_lock:
            if is_recording:
                # Currently recording, stop it
                is_recording = False
                print("F11 pressed. Signalling screen recording to stop...")
                # The screen_recording_thread will detect is_recording = False and exit its loop
                # Then it will release the video_writer
            else:
                # Not recording, start a new session
                is_recording = True
                keyboard_events.clear() # Clear events from previous sessions
                screen_thread_instance = threading.Thread(target=record_screen_loop, daemon=True) # Make daemon thread
                screen_thread_instance.start()
                print("F11 pressed. Starting new recording...")
        return # Don't process F11 as a regular key event

    if is_recording: # Only record keys if currently recording
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        keyboard_events.append(f"{timestamp}: Key {key} released")


if __name__ == "__main__":
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release, daemon=True) # Make daemon thread
    keyboard_listener.start()
    
    print("Press F11 to start/stop recording.")
    print("Press Ctrl+C to exit the application gracefully.")

    try:
        # Keep the main thread alive indefinitely to allow keyboard listener to run
        while True:
            time.sleep(0.1) # Use a small sleep to keep main thread alive and allow interrupt
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting application.")
        with recording_state_lock:
            is_recording = False # Ensure recording stops if active
        # The current_video_writer.release() and save_keyboard_events() logic
        # are handled by the record_screen_loop when is_recording becomes False
        if keyboard_listener.is_alive():
            keyboard_listener.stop() # Stop the keyboard listener thread if it's still running
        if screen_thread_instance and screen_thread_instance.is_alive():
            screen_thread_instance.join() # Wait for screen thread to finish
        print("Application exited.")
