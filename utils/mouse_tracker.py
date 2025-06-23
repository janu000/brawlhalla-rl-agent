from pynput.mouse import Controller
import mss
import time
import os

mouse = Controller()
sct = mss.mss()

def get_current_monitor(mouse_x, mouse_y):
    for monitor in sct.monitors[1:]:  # Skip [0] because it is the full virtual screen
        if (monitor['left'] <= mouse_x < monitor['left'] + monitor['width'] and
            monitor['top'] <= mouse_y < monitor['top'] + monitor['height']):
            return monitor
    return None

try:
    while True:
        pos = mouse.position  # Global position
        monitor = get_current_monitor(*pos)

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Global position: X={pos[0]} Y={pos[1]}")

        if monitor:
            rel_x = pos[0] - monitor["left"]
            rel_y = pos[1] - monitor["top"]
            print(f"Monitor relative position: X={rel_x}, Y={rel_y}")
        else:
            print("Mouse not on any defined monitor.")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopped.")
