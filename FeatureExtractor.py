import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import *

class CNNWithActionEmbedding(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_out_dim=256):
        super().__init__(observation_space, features_dim=1)  # temp

        self.cnn = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Get CNN output dim dynamically
        with th.no_grad():
            sample_input = th.zeros(1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
            cnn_out_dim = self.cnn(sample_input).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU()
        )

        # Action history embedding
        self.action_embedding = nn.Embedding(NUM_ACTIONS, EMBED_DIM)
        self.history_ff = nn.Sequential(
            nn.Linear(HISTORY_LENGTH * EMBED_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.final_dim = cnn_out_dim + 128
        self._features_dim = self.final_dim

    def forward(self, obs):
        x_img = obs["image"].float() / 255.0
        x_action = obs["action_history"].long()

        cnn_out = self.cnn(x_img)
        # cnn_feat = self.cnn_fc(cnn_out)

        embedded = self.action_embedding(x_action)  # shape: [B, N, D]
        embedded = embedded.view(embedded.size(0), -1)
        action_feat = self.history_ff(embedded)  # flatten to [batch, history_len * embed_dim]

        return th.cat([cnn_out, action_feat], dim=1)
