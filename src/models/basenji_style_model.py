import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class BasenjiDilatedBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.15):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout1d(dropout)
    def forward(self, x):
        h = F.gelu(self.norm(x))
        h = self.conv(h)
        h = self.drop(h)
        return x + h

class BasenjiStyle(BaseModel):
    """Lightweight Basenji/Basenji2-style dilated CNN model."""
    def __init__(self, input_length=256, channels=128, dilations=(1,2,4,8,16,32), dropout=0.15, hidden_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, channels, kernel_size=1),
            nn.BatchNorm1d(channels), nn.GELU()
        )
        self.blocks = nn.Sequential(*[BasenjiDilatedBlock(channels, d, dropout=dropout) for d in dilations])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x)))).squeeze(-1)
