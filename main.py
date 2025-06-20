from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
import time

from BrawlhallaEnv import BrawlhallaEnv
from FeatureExtractor import CNNWithActionEmbedding
from config import *

class ActionEmbeddingPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CNNWithActionEmbedding,
            features_extractor_kwargs=dict(cnn_out_dim=256),
            **kwargs,
        )

env = BrawlhallaEnv()
check_env(env, warn=True)

model = PPO(
    ActionEmbeddingPolicy,
    env,
    verbose=1,
    tensorboard_log="./ppo_brawlhalla_logs",
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS
)

print("Enter Brawhalla within 2s")
time.sleep(2)
print("Starting Training")

try:
    model.learn(total_timesteps=10_000)
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Closing environment...")
finally:
    env.close()

obs, _ = model.env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    print("Predicted action:", action)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated:
        obs = env.reset()