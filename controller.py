from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController, Button

import time
from config import ACTION_LUT

keyboard = Controller()
mouse = MouseController()

# Mapping for keys that need special handling (e.g., 'alt', 'ctrl', 'shift')
SPECIAL_KEYS = {
    'jump': Key.space,
    'light': 'j',
    'heavy': 'k',
    'throw': 'u', 
    'dodge': Key.shift_l, 
    'left': 'a',
    'right': 'd',
    'up': 'w',
    'down': 's',
    'idle': None
}

# Populate the main ACTION_MAPPER based on ACTION_LUT and SPECIAL_KEYS
# This will map action index to a list of pynput key objects or strings
ACTION_MAPPER = {}
for i, action_set in enumerate(ACTION_LUT):
    mapped_keys = []
    for action_name in action_set:
        if action_name in SPECIAL_KEYS:
            mapped_keys.append(SPECIAL_KEYS[action_name])
        else:
            # Handle combinations like 'left+jump' -> split and map
            sub_actions = action_name.split('+')
            for sub_action in sub_actions:
                if sub_action in SPECIAL_KEYS:
                    mapped_keys.append(SPECIAL_KEYS[sub_action])
                else:
                    # Default to string if not special key and not in SPECIAL_KEYS
                    mapped_keys.append(sub_action)
    ACTION_MAPPER[i] = mapped_keys


active_keys = set()

def press_keys(keys):
    global active_keys
    if not isinstance(keys, list):
        keys = [keys]
    for key in keys:
        if key is not None:
            keyboard.press(key)
            active_keys.add(key)
            time.sleep(0.005)

def release_keys(keys):
    global active_keys
    if not isinstance(keys, list):
        keys = [keys]
    for key in keys:
        if key is not None:
            keyboard.release(key)
            active_keys.discard(key)  # discard avoids KeyError if key not in set
            time.sleep(0.005)

def get_active_keys():
    return list(active_keys)

def release_all_keys():
    release_keys(list(SPECIAL_KEYS.values()))

def execute_action(action_idx, press_duration=0.05):
    keys_to_press = ACTION_MAPPER.get(action_idx, [])

    if keys_to_press:
        press_keys(keys_to_press)
        time.sleep(press_duration)
        release_keys(keys_to_press)
    elif action_idx == ACTION_LUT.index(['idle']):
        time.sleep(press_duration) # Still respect duration for idle

def click_at(x, y, button=Button.left):
    mouse.position = (x, y)
    time.sleep(0.1)  # Small delay to ensure mouse moved
    mouse.click(button)
    time.sleep(0.005)

def reset_o_health():
    click_at(2676, 50) # open settings
    time.sleep(0.2)
    click_at(3647, 113) # open bot section
    click_at(3670, 665) # set health
    click_at(3523, 800) # confirm and close

if __name__ == "__main__":
    print("Testing controller functions with ACTION_LUT...")
    time.sleep(1) # Give a moment before starting tests

    print(f"Initial active keys: {get_active_keys()}")

    print("Executing 'jump' (spacebar) for 0.5s...")
    execute_action(ACTION_LUT.index(['jump']), press_duration=0.5)
    print(f"Active keys after jump: {get_active_keys()}")
    time.sleep(1)

    print("Executing 'right' for 0.5s...")
    execute_action(ACTION_LUT.index(['right']), press_duration=0.5)
    print(f"Active keys after right: {get_active_keys()}")
    time.sleep(1)

    print("Executing 'left+light' for 0.3s...")
    execute_action(ACTION_LUT.index(['left+light']), press_duration=0.3)
    print(f"Active keys after left+light: {get_active_keys()}")
    time.sleep(1)

    print("Executing 'idle' for 0.2s...")
    execute_action(ACTION_LUT.index(['idle']), press_duration=0.2)
    print(f"Active keys after idle: {get_active_keys()}")
    time.sleep(1)

    print("Executing 'jump+right' for 0.2s...")
    execute_action(ACTION_LUT.index(['jump+right']), press_duration=0.2)
    print(f"Active keys after idle: {get_active_keys()}")
    time.sleep(1)

    print("Controller tests complete.")
