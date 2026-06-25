import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, seq_length=30, num_features=12):
        super(Autoencoder, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        flattened_dim = seq_length * num_features
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, flattened_dim)
        )

    def forward(self, x):
        # x is (Batch, 30, 12)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        return x_recon
