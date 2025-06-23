from pynput.keyboard import Controller, Key, KeyCode, Listener
from pynput.mouse import Controller as MouseController, Button
import time
import numpy as np



class BrawlhallaController:
    """
    Handles keyboard and mouse input for controlling Brawlhalla via RL agent actions.
    Provides action mapping, key management, and utility functions for game automation.
    """
    def __init__(self):
        """
        Initialize keyboard/mouse controllers, action mappings, and key listeners.
        """
        self.keyboard = Controller()
        self.mouse = MouseController()

        self.ACTION_LUT = [
            'idle',
            'up+dodge', 'down+dodge', 'left+dodge', 'right+dodge', 'dodge',
            'left+jump', 'right+jump', 'jump',
            'up+throw', 'down+throw', 'left+throw', 'right+throw',
            'down+light', 'up+light', 'left+light', 'right+light',
            'up+heavy', 'down+heavy', 'left+heavy', 'right+heavy',
            'down', 'left', 'right', 
        ]

        self.SPECIAL_KEYS = {
            'dodge': Key.shift_l,
            'jump': Key.space,
            'light': 'j',
            'heavy': 'k',
            'throw': 'u', 
            'left': 'a',
            'right': 'd',
            'up': 'w',
            'down': 's',
            'idle': None
        }

        self.name_to_keys = {}
        self.idx_to_keys = {}
        self.reverse_action_mapper = {}
        self._build_action_mappings()
        self.extended_keys = self._build_extended_keys()
        self.action_delay = 0.001
        self.pressed_keys = set()
        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _build_action_mappings(self):
        """
        Build mappings from action names/indices to key lists and reverse.
        """
        for idx, action_str in enumerate(self.ACTION_LUT):
            sub_actions = action_str.split('+')
            key_list = []
            for sub_action in sub_actions:
                key = self.SPECIAL_KEYS.get(sub_action)
                if key is None:
                    continue
                if isinstance(key, str):
                    key = KeyCode.from_char(key)
                key_list.append(key)
            self.name_to_keys[action_str] = key_list
            self.idx_to_keys[idx] = key_list
            self.reverse_action_mapper[tuple(sorted(key_list, key=lambda x: str(x)))] = action_str

    def _build_extended_keys(self):
        """
        Build a list of all special keys, including uppercase variants for letters.
        """
        extended_keys = []
        for value in self.SPECIAL_KEYS.values():
            if isinstance(value, str):
                extended_keys.append(KeyCode.from_char(value.lower()))
                extended_keys.append(KeyCode.from_char(value.upper()))
            elif value is not None:
                extended_keys.append(value)
        return extended_keys

    def _on_press(self, key):
        """
        Listener callback for key press events. Handles shift+char for uppercase.
        """
        if isinstance(key, KeyCode) and Key.shift in self.pressed_keys:
            char = key.char
            if char and char.isalpha():
                key = KeyCode.from_char(char.upper())
        self.pressed_keys.add(key)

    def _on_release(self, key):
        """
        Listener callback for key release events.
        """
        self.pressed_keys.discard(key)

    def press_keys(self, keys):
        """
        Press the specified keys if not already pressed.
        """
        if isinstance(keys, (str, int, KeyCode)) or not hasattr(keys, '__iter__'):
            keys = [keys]
        for key in keys:
            if key is not None and key not in self.pressed_keys:
                self.keyboard.press(key)
                time.sleep(self.action_delay)

    def release_keys(self, keys):
        """
        Release the specified keys if pressed.
        Handles both lower and upper case for KeyCode.
        """
        if isinstance(keys, (str, int)) or not hasattr(keys, '__iter__'):
            keys = [keys]
        for key in keys:
            if key is not None:
                if isinstance(key, KeyCode):
                    char = key.char
                    if char is not None:
                        lower_key = KeyCode.from_char(char.lower())
                        upper_key = KeyCode.from_char(char.upper())
                        if lower_key in self.pressed_keys:
                            if Key.shift in self.pressed_keys:
                                self.keyboard.release(upper_key)
                            else:
                                self.keyboard.release(lower_key)
                        elif upper_key in self.pressed_keys:
                            self.keyboard.release(upper_key)
                    else:
                        self.keyboard.release(key)
                else:
                    self.keyboard.release(key)
                time.sleep(self.action_delay)

    def get_active_keys(self):
        """
        Return a list of currently pressed keys.
        """
        return list(self.pressed_keys.copy())

    def release_all_keys(self, unjam=False):
        """
        Release all currently pressed keys. Optionally 'unjam' by pressing and releasing all extended keys.
        """
        self.release_keys(self.pressed_keys.copy())
        if unjam:
            for key in self.extended_keys:
                self.keyboard.press(key)
                time.sleep(self.action_delay)
                self.keyboard.release(key)
                time.sleep(self.action_delay)

    def execute_action(self, action_idx):
        """
        Execute the action corresponding to the given index by pressing/releasing the appropriate keys.
        """
        if not isinstance(action_idx, (int, np.integer)):
            raise TypeError(f"Expected action_idx to be an int or np.integer, but got {type(action_idx).__name__}")
        keys_to_press = set(self.idx_to_keys[action_idx])
        keys_to_release = self.pressed_keys - keys_to_press
        self.release_keys(keys_to_release)
        self.press_keys(keys_to_press)

    def click_at(self, x, y, button=Button.left, sleep=0.005):
        """
        Move the mouse to (x, y) and click the specified button.
        """
        self.mouse.position = (x, y)
        time.sleep(0.1)
        self.mouse.click(button)
        time.sleep(sleep)

    def reset_o_health(self):
        """
        Automate mouse clicks to reset opponent health in the game UI.
        """
        self.click_at(2676, 50, sleep=0.2)
        self.click_at(3647, 113, sleep=0.1)
        self.click_at(3670, 665, sleep=0.1)
        self.click_at(3523, 800, sleep=0.1)

if __name__ == "__main__":
    print("Testing controller functions with ACTION_LUT...")
    manager = BrawlhallaController()
    time.sleep(1)
    key_a = KeyCode.from_char('a')
    key_A = KeyCode.from_char('A')
    print(f"'a' KeyCode: {key_a} | char: {key_a.char}")
    print(f"'A' KeyCode: {key_A} | char: {key_A.char}")
    manager.press_keys(['a', Key.shift, 'a'])
    manager.release_keys(['a', 'A', Key.shift])
    print(f"Active keys: {manager.get_active_keys()}")
    manager.release_all_keys(unjam=True)
    print("Controller tests complete.")
    manager.listener.stop()
