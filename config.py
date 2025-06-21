IMAGE_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 144

EMBED_DIM = 16
LEARNING_RATE = 1e-4
REC_FPS = 60
LEARNING_FPS = 30
LSTM_HIDDEN_SIZE = 128

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

# PPO hyperparameters
N_STEPS = 2048         # Number of environment steps per training iteration
BATCH_SIZE = 256       # Size of each minibatch for training
N_EPOCHS = 8           # Number of times to iterate over the collected data

HEALTH1_POS = (100, 1774) # Note: y, x !
HEALTH2_POS = (100, 1867) # Note: y, x !