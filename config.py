IMAGE_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 144
HISTORY_LENGTH = 8      # last N actions
EMBED_DIM = 16
LEARNING_RATE = 1e-3 

ACTION_LUT = [
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
NUM_ACTIONS = len(ACTION_LUT)
