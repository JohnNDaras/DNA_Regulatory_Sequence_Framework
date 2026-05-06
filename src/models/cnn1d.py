import torch.nn as nn
from .base_model import BaseModel
class CNN1D(BaseModel):
    def __init__(self, in_channels=4, channels=(32,64), kernel_sizes=(15,5), pool_size=4, hidden_dim=64, dropout=0.2, input_length=256):
        super().__init__()
        layers = []
        c_in = in_channels
        length = input_length
        for c_out, k in zip(channels, kernel_sizes):
            layers += [nn.Conv1d(c_in, c_out, kernel_size=k), nn.BatchNorm1d(c_out), nn.Dropout1d(dropout), nn.ELU(inplace=True), nn.MaxPool1d(pool_size)]
            length = (length - k + 1) // pool_size
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c_in * length, hidden_dim), nn.Dropout(dropout), nn.ELU(inplace=True), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        x = self.conv(x)
        return self.head(x.view(x.size(0), -1)).squeeze(-1)
