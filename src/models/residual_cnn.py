import torch.nn as nn
from .base_model import BaseModel
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=p),
            nn.BatchNorm1d(channels), nn.ELU(inplace=True), nn.Dropout1d(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=p), nn.BatchNorm1d(channels)
        )
        self.act = nn.ELU(inplace=True)
    def forward(self, x):
        return self.act(self.net(x) + x)
class ResidualCNN(BaseModel):
    def __init__(self, input_length=256, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, 32, kernel_size=15), nn.BatchNorm1d(32), nn.ELU(inplace=True), nn.MaxPool1d(4))
        length = (input_length - 15 + 1) // 4
        self.res = nn.Sequential(ResidualBlock1D(32, 5, dropout), ResidualBlock1D(32, 5, dropout), nn.MaxPool1d(4))
        length = length // 4
        self.head = nn.Sequential(nn.Linear(32 * length, hidden_dim), nn.ELU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        return self.head(x.view(x.size(0), -1)).squeeze(-1)
