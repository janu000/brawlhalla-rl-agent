config = {
    "IMAGE_CHANNELS": 3,
    "IMAGE_WIDTH": 256,
    "IMAGE_HEIGHT": 144,

    "EMBED_DIM": 16,
    "LEARNING_RATE": 3e-4,
    "REC_FPS": 60,
    "LEARNING_FPS": 20, # steps per second of the agent (the actual game runs continuously). Training fps may differ due to respawn and policy update durations
    "MAX_EPS_LEN": 1000,
    "LSTM_HIDDEN_SIZE": 128,

    # PPO hyperparameters
    "N_STEPS": 1024,       # Number of environment steps per training iteration
    "BATCH_SIZE": 512,       # Size of each minibatch for training
    "N_EPOCHS": 6,           # Number of times to iterate over the collected data

    # Adjust to your monitor and game setup
    "HEALTH_POS1": (100, 1774), # Note: y, x !
    "HEALTH_POS2": (96, 1875), # Note: y, x !
    "GAME_MONITOR_ID": 1,
}