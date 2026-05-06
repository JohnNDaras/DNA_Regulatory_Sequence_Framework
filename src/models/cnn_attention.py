import torch.nn as nn
from .base_model import BaseModel
class CNNAttention(BaseModel):
    def __init__(self, input_length=256, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4,32,kernel_size=15), nn.BatchNorm1d(32), nn.ELU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(32,64,kernel_size=5), nn.BatchNorm1d(64), nn.ELU(inplace=True)
        )
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(nn.Linear(64, hidden_dim), nn.ELU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.head(x).squeeze(-1)
