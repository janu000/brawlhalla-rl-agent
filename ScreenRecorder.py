import threading
import time
from collections import deque
import mss
import numpy as np
import cv2
from config import IMAGE_CHANNELS

class ScreenRecorder:
    def __init__(self, fps=30, region=None, buffer_size=30, render=False, monitor_id=1, resolution = (1920, 1080), grayscale=False):
        self.region = region  # e.g., {"top": 0, "left": 0, "width": 1366, "height": 768}
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps = fps
        self.render = render
        self.monitor_id = monitor_id
        self.resolution = resolution
        self.grayscale = True if IMAGE_CHANNELS == 1 else False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.daemon = True  # Will close with main thread
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _record_loop(self):
        with mss.mss() as sct:
            monitor = self.region if self.region else sct.monitors[self.monitor_id]
            disp_monitor = sct.monitors[2 if (self.monitor_id == 1) else 1]
            target_frame_duration = 1.0 / self.fps

            while self.running:
                start_time = time.monotonic()
                frame = np.array(sct.grab(monitor))[:, :, :3]  # Returns BGR image after dropping alpha

                with self.lock:
                    self.buffer.append(frame)

                if self.render:
                    display_width = int(disp_monitor['width'] * 0.3)
                    display_height = int(disp_monitor['height'] * 0.3)

                    if self.grayscale:
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    else:
                        display_frame = frame

                    resized_frame = cv2.resize(frame, (self.resolution)) # resize to set resolution
                    resized_frame = cv2.resize(display_frame, (display_width, display_height)) # resize to display window
                    cv2.imshow("Live Recording", resized_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False # Allow graceful exit from live view

                end_time = time.monotonic()
                elapsed_time = end_time - start_time
                sleep_time = target_frame_duration - elapsed_time

                if sleep_time > 0:
                    time.sleep(sleep_time)


        cv2.destroyAllWindows()

    def get_latest_frames(self, count):
        with self.lock:
            return list(self.buffer)[-count:] if count <= len(self.buffer) else list(self.buffer)

    def get_latest_frame(self):
        with self.lock:
            return self.buffer[-1] if self.buffer else None
