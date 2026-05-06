import torch.nn as nn
from .base_model import BaseModel

class DanQStyle(BaseModel):
    """Lightweight DanQ-inspired Conv + BiLSTM model."""
    def __init__(self, input_length=256, conv_channels=128, lstm_hidden=128, dropout=0.2, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, conv_channels, kernel_size=15),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout1d(dropout),
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * lstm_hidden, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        h, _ = self.lstm(x)
        h = self.dropout(h)
        h = h.amax(dim=1)
        return self.head(h).squeeze(-1)
