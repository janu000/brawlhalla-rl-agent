# Dummy Classes for the BrawlhallaEnv during offline training

class BrawlhallaController:
    """
    Dummy controller for the BrawlhallaEnv when Keyboard and Mouse actions are not needed
    """
    def __init__(self):
        self.ACTION_LUT = [
            'idle',
            'up+dodge', 'down+dodge', 'left+dodge', 'right+dodge', 'dodge',
            'left+jump', 'right+jump', 'jump',
            'up+throw', 'down+throw', 'left+throw', 'right+throw',
            'down+light', 'up+light', 'left+light', 'right+light',
            'up+heavy', 'down+heavy', 'left+heavy', 'right+heavy',
            'down', 'left', 'right', 
        ]

        self.name_to_keys = {}
        self.idx_to_keys = {}
        self.reverse_action_mapper = {}
        self.extended_keys = self._build_extended_keys()
        self.pressed_keys = set()

    def _build_extended_keys(self):       
        return []

    def press_keys(self, keys):
        pass

    def release_keys(self, keys):
        pass

    def get_active_keys(self):
        return []

    def release_all_keys(self, unjam=False):
        pass

    def execute_action(self, action_idx):
        pass

    def click_at(self, x, y, button=None, sleep=0.005):
        pass

    def reset_o_health(self):
        pass


class ScreenRecorder:
    def __init__(self, fps=30, region=None, buffer_size=30, render=False, monitor_id=1, resolution = (1920, 1080), grayscale=False):
        self.running = False
        self.thread = None
        self.fps = fps
        self.render = render
        self.monitor_id = monitor_id
        self.resolution = resolution
        self.grayscale = grayscale

    def start(self):
        pass

    def stop(self):
        pass

    def get_latest_frames(self, count):
        return []

    def get_latest_frame(self):
        return None
