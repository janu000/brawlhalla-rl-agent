import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import *

class ActionEmbeddingRecurrentPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CNNWithActionEmbedding,
            **kwargs,
        )

       
class CNNWithActionEmbedding(BaseFeaturesExtractor):
    def __init__(self, observation_space, out_dim=128, num_actions=1):
        super().__init__(observation_space, features_dim=1)  # temp

        self.cnn = nn.Sequential(
            # Conv1: 3 -> 16 channels, small kernel + stride 2 for downsample
            nn.Conv2d(IMAGE_CHANNELS, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Conv2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4: 64 -> 128 channels, no downsample
            nn.Conv2d(64, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            # Global avg pool
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim-EMBED_DIM),
            nn.ReLU()
        )

        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, EMBED_DIM)

        self.final_dim = out_dim
        self._features_dim = self.final_dim

    def forward(self, obs):
        # Process last frame
        x_img = obs["image"]
        cnn_out = self.cnn(x_img)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_feat = self.fc(cnn_out)

        # Process last executed action
        x_action = obs["last_executed_action"].squeeze(-1).long()  # shape: [B]
        action_feat = self.action_embedding(x_action)  # shape: [B, EMBED_DIM]

        # Combine image and action features
        return th.cat([cnn_feat, action_feat], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class CNNWithResiduals(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        return x