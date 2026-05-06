from .logistic_baseline import LogisticBaseline
from .mlp_baseline import MLPBaseline
from .cnn1d import CNN1D
from .residual_cnn import ResidualCNN
from .cnn_attention import CNNAttention
from .rc_dilated_bigru_gated import RCDilatedCNNBiGRUGated
from .deepsea_model import DeepSEAStyle
from .danq_model import DanQStyle
from .basenji_style_model import BasenjiStyle

def build_model(model_cfg: dict, input_length: int):
    name = model_cfg["name"]
    if name == "logistic_baseline": return LogisticBaseline(seq_len=input_length)
    if name == "mlp_baseline": return MLPBaseline(seq_len=input_length, hidden_dim=model_cfg.get("hidden_dim", 128))
    if name == "cnn1d": return CNN1D(input_length=input_length)
    if name == "residual_cnn": return ResidualCNN(input_length=input_length)
    if name == "cnn_attention": return CNNAttention(input_length=input_length)
    if name == "deepsea_style": return DeepSEAStyle(input_length=input_length)
    if name == "danq_style": return DanQStyle(input_length=input_length)
    if name == "basenji_style": return BasenjiStyle(input_length=input_length)
    if name == "rc_dilated_bigru_gated": return RCDilatedCNNBiGRUGated(seq_len=input_length)
    raise ValueError(name)
