import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

def reverse_complement_batch(x):
    return x[:, [3, 2, 1, 0], :].flip(dims=[-1]).clone()

class SpatialDropout1D(nn.Dropout2d):
    def forward(self, x):
        return super().forward(x.unsqueeze(3)).squeeze(3)

class SE1D(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        hid = max(1, channels // r)
        self.fc1 = nn.Conv1d(channels, hid, 1)
        self.fc2 = nn.Conv1d(hid, channels, 1)
    def forward(self, x):
        z = x.mean(-1, keepdim=True)
        s = F.elu(self.fc1(z), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class DilatedResBlock(nn.Module):
    def __init__(self, ch, k=7, d=1, p_drop=0.1):
        super().__init__()
        pad = ((k - 1) // 2) * d
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.drop = SpatialDropout1D(p_drop)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=pad, dilation=d, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.se = SE1D(ch, r=8)
    def forward(self, x):
        h = F.elu(self.bn1(self.conv1(x)), inplace=True)
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.elu(h + x, inplace=True)

class AttnPool1D(nn.Module):
    def __init__(self, ch, temperature=1.0):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1, bias=True)
        self.temperature = temperature
    def forward(self, x):
        a = torch.softmax(self.score(x) / self.temperature, dim=-1)
        return (x * a).sum(-1)

class RCDilatedCNNBiGRUGated(BaseModel):
    def __init__(self, seq_len=256, stem_channels=64, block_channels=128, n_blocks=6,
                 kernel_size=7, dilations=(1,2,4,8,16,32), rnn_hidden=128, rnn_layers=1,
                 p_drop=0.15, n_hidden=256, n_output_channels=1, rc_fusion="lse", lse_temp=2.0):
        super().__init__()
        self.seq_len = seq_len
        self.lse_temp = lse_temp
        self.rc_fusion = rc_fusion
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_channels, kernel_size=19, padding=9, bias=False),
            nn.BatchNorm1d(stem_channels), nn.ELU(inplace=True), SpatialDropout1D(p_drop)
        )
        self.proj = nn.Sequential(
            nn.Conv1d(stem_channels, block_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(block_channels), nn.ELU(inplace=True)
        )
        self.blocks = nn.Sequential(*[
            DilatedResBlock(block_channels, k=kernel_size, d=d, p_drop=p_drop)
            for d in dilations[:n_blocks]
        ])
        self.bigru = nn.GRU(input_size=block_channels, hidden_size=rnn_hidden,
                            num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.rnn_drop = nn.Dropout(p_drop)
        self.rnn_norm = nn.LayerNorm(2 * rnn_hidden)
        rnn_out_ch = 2 * rnn_hidden
        self.attn = AttnPool1D(rnn_out_ch)
        self.head = nn.Sequential(
            nn.Linear(3 * rnn_out_ch, n_hidden),
            nn.ELU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(n_hidden, n_output_channels)
        )
    def _features(self, x):
        h = self.stem(x)
        h = self.proj(h)
        h = self.blocks(h)
        return h
    def _fuse(self, fwd, rc):
        if self.rc_fusion == "max":
            return torch.maximum(fwd, rc)
        if self.rc_fusion == "mean":
            return 0.5 * (fwd + rc)
        t = self.lse_temp
        return torch.logsumexp(torch.stack([fwd / t, rc / t], dim=0), dim=0) * t
    def _gated_attention_refine(self, h_seq):
        scale = h_seq.size(-1) ** 0.5
        sim = torch.bmm(h_seq, h_seq.transpose(1, 2)) / scale
        gate = torch.sigmoid(sim)
        refined = torch.bmm(gate, h_seq) + h_seq
        return self.rnn_norm(refined)
    def forward(self, x):
        fwd = self._features(x)
        rc = self._features(reverse_complement_batch(x))
        h = self._fuse(fwd, rc)
        h_seq = h.transpose(1, 2)
        h_seq, _ = self.bigru(h_seq)
        h_seq = self.rnn_drop(h_seq)
        h_seq = self.rnn_norm(h_seq)
        h_seq = self._gated_attention_refine(h_seq)
        h_seq = h_seq.transpose(1, 2)
        h_attn = self.attn(h_seq)
        h_avg = h_seq.mean(-1)
        h_max = h_seq.amax(-1)
        return self.head(torch.cat([h_attn, h_avg, h_max], dim=1)).squeeze(-1)
