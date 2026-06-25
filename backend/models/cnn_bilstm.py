import torch
import torch.nn as nn

class CNNBiLSTM(nn.Module):
    def __init__(self, num_features=12, seq_length=30):
        super(CNNBiLSTM, self).__init__()
        # Input shape expected by PyTorch Conv1d: (Batch, Channels, Length)
        # We handle permutation in the forward pass so it accepts (Batch, Length, Features)
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)
        
        self.dropout1 = nn.Dropout(p=0.2)
        
        # BiLSTM expects (Batch, Seq_len, Input_Size) because batch_first=True
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        
        # Unidirectional LSTM taking the full sequence output from BiLSTM (64 * 2 = 128 dim)
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, batch_first=True, bidirectional=False)
        
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Output Heads (10th, 50th, 90th percentiles)
        self.head_10 = nn.Linear(32, 1)
        self.head_50 = nn.Linear(32, 1)
        self.head_90 = nn.Linear(32, 1)

    def forward(self, x):
        # x is (Batch, Timesteps, Features)
        x = x.permute(0, 2, 1) # (Batch, Features, Timesteps)
        
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        
        # Layer 3
        x = self.dropout1(x)
        
        # Permute back for LSTM: (Batch, Channels, Length) -> (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        
        # Layer 4: BiLSTM (returns full sequence)
        x, _ = self.bilstm(x)
        
        # Layer 5: Unidirectional LSTM (returns only final hidden state)
        _, (h_n, _) = self.lstm(x)
        
        # Extract the final hidden state of the top LSTM layer
        # h_n shape: (num_layers * num_directions, Batch, Hidden_size) -> (1, Batch, 32)
        x = h_n[-1] # shape: (Batch, 32)
        
        # Layer 6
        x = self.dropout2(x)
        
        # Output Heads
        out_10 = self.head_10(x)
        out_50 = self.head_50(x)
        out_90 = self.head_90(x)
        
        return out_10, out_50, out_90
