import torch.nn as nn
from .base_model import BaseModel
class MLPBaseline(BaseModel):
    def __init__(self, seq_len: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * seq_len, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.net(x.view(x.size(0), -1)).squeeze(-1)
