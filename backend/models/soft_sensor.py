import torch
import torch.nn as nn

class SoftSensor(nn.Module):
    def __init__(self, in_features=14):
        super(SoftSensor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x is expected to be (Batch, 14)
        return self.net(x)
