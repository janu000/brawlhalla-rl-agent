IMAGE_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 144

EMBED_DIM = 16
LEARNING_RATE = 1e-4
REC_FPS = 60
LEARNING_FPS = 20
LSTM_HIDDEN_SIZE = 128

# PPO hyperparameters
N_STEPS = 1024         # Number of environment steps per training iteration
BATCH_SIZE = 512       # Size of each minibatch for training
N_EPOCHS = 6           # Number of times to iterate over the collected data

HEALTH1_POS = (100, 1774) # Note: y, x !
HEALTH2_POS = (100, 1867) # Note: y, x !
GAME_MONITOR_ID = 1