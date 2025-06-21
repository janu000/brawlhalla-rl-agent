import keyboard as kb
from pynput.keyboard import Controller, Key, Listener
from pynput.mouse import Controller as MouseController, Button
import time

class BrawlhallaController:
    def __init__(self):
        self.keyboard = Controller()
        self.mouse = MouseController()

        self.ACTION_LUT = [
            ['left'],
            ['down'],
            ['right'],
            ['jump'],
            ['dodge'],
            ['idle'],

            ['up+dodge'],
            ['up+light'],
            ['up+heavy'],
            ['up+throw'],

            ['left+jump'],
            ['left+dodge'],
            ['left+light'],
            ['left+heavy'],
            ['left+throw'],

            ['down+dodge'],
            ['down+light'],
            ['down+heavy'],
            ['down+throw'],

            ['right+jump'],
            ['right+dodge'],
            ['right+light'],
            ['right+heavy'],
            ['right+throw'],
        ]

        self.SPECIAL_KEYS = {
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
        
        self.ACTION_MAPPER = {}
        for i, action_set in enumerate(self.ACTION_LUT):
            mapped_keys = []
            for action_name in action_set:
                if action_name in self.SPECIAL_KEYS:
                    mapped_keys.append(self.SPECIAL_KEYS[action_name])
                else:
                    sub_actions = action_name.split('+')
                    for sub_action in sub_actions:
                        if sub_action in self.SPECIAL_KEYS:
                            mapped_keys.append(self.SPECIAL_KEYS[sub_action])
                        else:
                            mapped_keys.append(sub_action)
            self.ACTION_MAPPER[i] = mapped_keys

        self.pressed_keys = set()
        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        self.pressed_keys.add(key)

    def _on_release(self, key):
        self.pressed_keys.discard(key)

    def press_keys(self, keys):
        if isinstance(keys, (str, int)) or not hasattr(keys, '__iter__'):
            keys = [keys]
        for key in keys:
            if key is not None:
                self.keyboard.press(key)
                time.sleep(0.005)

    def release_keys(self, keys):
        if isinstance(keys, (str, int)) or not hasattr(keys, '__iter__'):
            keys = [keys]
        for key in keys:
            if key is not None:
                if isinstance(key, str) and key.isalpha() and key.isupper():
                    self.keyboard.release(key.lower())
                    self.keyboard.release(Key.shift)
                else:
                    self.keyboard.release(key)

                time.sleep(0.005)

    def get_active_keys(self):
        return list(self.pressed_keys)

    def release_all_keys(self):
        count = 0
        while self.pressed_keys != set([]):
            if count > 0:
                print("still pressed", self.pressed_keys)
            self.release_keys(self.pressed_keys.copy())
            time.sleep(0.05)

            count += 1
            if count > 20:
                raise RuntimeError(("Failed to release keys: " + str(self.pressed_keys)))

    def execute_action(self, action_idx, press_duration=0.05):
        keys_to_press = self.ACTION_MAPPER.get(action_idx, [])

        if keys_to_press:
            self.press_keys(keys_to_press)
            time.sleep(press_duration)
            self.release_keys(keys_to_press)
        elif action_idx == self.ACTION_LUT.index(['idle']):
            time.sleep(press_duration)

    def click_at(self, x, y, button=Button.left, sleep=0.005):
        self.mouse.position = (x, y)
        time.sleep(0.1)
        self.mouse.click(button)
        time.sleep(sleep)

    def reset_o_health(self):
        self.click_at(2676, 50, sleep=0.2)
        self.click_at(3647, 113, sleep=0.1)
        self.click_at(3670, 665, sleep=0.1)
        self.click_at(3523, 800, sleep=0.1)

if __name__ == "__main__":
    print("Testing controller functions with ACTION_LUT...")
    manager = BrawlhallaController()
    time.sleep(1)

    print(f"Initial active keys: {manager.get_active_keys()}")

    print("Executing 'jump' (spacebar) for 0.5s...")
    manager.execute_action(manager.ACTION_LUT.index(['jump']), press_duration=0.5)
    print(f"Active keys after jump: {manager.get_active_keys()}")
    time.sleep(1)

    print("Executing 'right' for 0.5s...")
    manager.execute_action(manager.ACTION_LUT.index(['right']), press_duration=0.5)
    print(f"Active keys after right: {manager.get_active_keys()}")
    time.sleep(1)

    print("Executing 'left+light' for 0.3s...")
    manager.execute_action(manager.ACTION_LUT.index(['left+light']), press_duration=0.3)
    print(f"Active keys after left+light: {manager.get_active_keys()}")
    time.sleep(1)

    print("Executing 'idle' for 0.2s...")
    manager.execute_action(manager.ACTION_LUT.index(['idle']), press_duration=0.2)
    print(f"Active keys after idle: {manager.get_active_keys()}")
    time.sleep(1)

    print("Executing 'jump+right' for 0.2s...")
    manager.execute_action(manager.ACTION_LUT.index(['right+jump']), press_duration=0.2)
    print(f"Active keys after idle: {manager.get_active_keys()}")
    time.sleep(1)

    print("Controller tests complete.")

    manager.listener.stop()
