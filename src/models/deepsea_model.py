import torch.nn as nn
from .base_model import BaseModel

class DeepSEAStyle(BaseModel):
    """Lightweight DeepSEA-inspired CNN for controlled in-framework comparison."""
    def __init__(self, input_length=256, dropout=0.2, hidden_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=8), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(4), nn.Dropout1d(dropout),
            nn.Conv1d(64, 128, kernel_size=8), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(4), nn.Dropout1d(dropout),
            nn.Conv1d(128, 256, kernel_size=8), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)
