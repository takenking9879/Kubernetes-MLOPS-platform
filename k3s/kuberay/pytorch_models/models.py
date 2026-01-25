import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=28 * 28, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)
