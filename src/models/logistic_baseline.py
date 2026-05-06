import torch.nn as nn
from .base_model import BaseModel
class LogisticBaseline(BaseModel):
    def __init__(self, seq_len: int):
        super().__init__()
        self.linear = nn.Linear(4 * seq_len, 1)
    def forward(self, x):
        return self.linear(x.view(x.size(0), -1)).squeeze(-1)
