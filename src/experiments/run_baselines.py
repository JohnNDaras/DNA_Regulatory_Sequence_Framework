import os, json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.utils.config_utils import load_yaml
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.genome_loader import load_genome
from src.data.bed_loader import load_bed
from src.data.dataset import BedPeaksDataset, BedPeaksDatasetBetter
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.thresholding import select_best_threshold

def make_dirs(config):
    for key in ("checkpoints_dir", "figures_dir", "tables_dir", "logs_dir"):
        os.makedirs(config["paths"][key], exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

def load_task_data(config, test=False):
    task = config.get("task", "atac").lower()
    include = config["data"].get("include_chr_x_y", False)
    if task == "ctcf":
        df = load_bed(config["paths"]["ctcf_bed_path"], include_chr_x_y=include)
        train = df[df["chrom"].isin(["chr%i" % i for i in range(4, 23)])].copy()
        val = df[df["chrom"].isin(["chr2", "chr3"])].copy()
        tst = df[df["chrom"].isin(["chr1"])].copy()
        return (train, val, tst) if test else (train, val)
    df = load_bed(config["paths"]["atac_bed_path"], include_chr_x_y=include)
    val = df[df["chrom"].isin(["chr3", "chr4"])].copy()
    train = df[~df["chrom"].isin(["chr3", "chr4"])].copy()
    return (train, val, None) if test else (train, val)

def build_loader(df, genome, config, improved=False):
    L = config["data"]["context_length"]
    if improved:
        ds = BedPeaksDatasetBetter(
            df, genome, L,
            n_neg=config["data"].get("n_neg", 3),
            min_gap=config["data"].get("min_gap", 2000),
            max_tries=config["data"].get("max_tries", 20),
            gc_tol=config["data"].get("gc_tol", 0.05),
            max_N_frac=config["data"].get("max_N_frac", 0.1),
            rng_seed=config.get("seed", 42),
        )
    else:
        ds = BedPeaksDataset(df, genome, L)
    return DataLoader(ds, batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"])

def evaluate_model(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).float()
            logits = model(xb).squeeze()
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels_all.append(yb.numpy())
    import numpy as np
    return np.concatenate(probs_all), np.concatenate(labels_all).astype("float32")

from src.models.logistic_baseline import LogisticBaseline
from src.models.mlp_baseline import MLPBaseline
from src.models.cnn1d import CNN1D
from src.models.rc_dilated_bigru_gated import RCDilatedCNNBiGRUGated
from src.training.train_wrapper import train_model_boosted_iter
from src.evaluation.benchmark_table import build_benchmark_table
LOGGER = get_logger("baselines")

def build_models(config):
    L = config["data"]["context_length"]
    m = config["model"]
    return [
        {"name": "logistic_baseline", "model": LogisticBaseline(seq_len=L), "improved_negatives": False},
        {"name": "mlp_baseline", "model": MLPBaseline(seq_len=L, hidden_dim=128), "improved_negatives": False},
        {"name": "cnn1d_baseline", "model": CNN1D(input_length=L, channels=(32,64), kernel_sizes=(15,5), pool_size=4, hidden_dim=64, dropout=0.2), "improved_negatives": False},
        {"name": "rc_dilated_bigru_gated", "model": RCDilatedCNNBiGRUGated(seq_len=L, stem_channels=m.get("stem_channels",64), block_channels=m.get("block_channels",128), n_blocks=m.get("n_blocks",6), kernel_size=m.get("kernel_size",7), dilations=tuple(m.get("dilations",[1,2,4,8,16,32])), rnn_hidden=m.get("rnn_hidden",128), rnn_layers=m.get("rnn_layers",1), p_drop=m.get("dropout",0.15), n_hidden=m.get("hidden_dim",256), n_output_channels=m.get("n_outputs",1), rc_fusion=m.get("rc_fusion","lse"), lse_temp=m.get("lse_temp",2.0)), "improved_negatives": True},
    ]

def run(config_path="configs/base_config.yaml"):
    config = load_yaml(config_path)
    set_seed(config.get("seed", 42))
    make_dirs(config)
    LOGGER.info("Loading genome...")
    genome = load_genome(config["paths"]["genome_path"])
    LOGGER.info("Loading train/val/test splits...")
    train_df, val_df, _ = load_task_data(config, test=True)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    results = []
    for entry in build_models(config):
        name = entry["name"]
        LOGGER.info(f"Running model: {name}")
        train_loader = build_loader(train_df, genome, config, improved=entry["improved_negatives"])
        val_loader = build_loader(val_df, genome, config, improved=False)
        model, train_accs, val_accs = train_model_boosted_iter(
            model=entry["model"], train_loader=train_loader, val_loader=val_loader,
            epochs=config["train"].get("epochs",20), steps_per_epoch=config["train"].get("steps_per_epoch",300),
            patience=config["train"].get("patience",6), base_lr=config["train"].get("lr",3e-3),
            verbose=True, device=device,
        )
        probs, labels = evaluate_model(model, val_loader, device)
        threshold = select_best_threshold(labels, probs)
        metrics = compute_classification_metrics(labels, probs, threshold=threshold)
        ckpt = os.path.join(config["paths"]["checkpoints_dir"], f"{name}.pt")
        torch.save(model.state_dict(), ckpt)
        with open(os.path.join(config["paths"]["logs_dir"], f"{name}_history.json"), "w") as f:
            json.dump({"train_accs":[float(x) for x in train_accs], "val_accs":[float(x) for x in val_accs]}, f, indent=2)
        results.append({"model": name, "val_accuracy": metrics["accuracy"], "val_f1": metrics["f1"], "val_auroc": metrics["auroc"], "val_auprc": metrics["auprc"], "threshold": float(threshold), "checkpoint": ckpt})
        with open(os.path.join(config["paths"]["tables_dir"], f"{name}_metrics.json"), "w") as f:
            json.dump({"validation": metrics}, f, indent=2)
    df = build_benchmark_table(results)
    out = os.path.join(config["paths"]["tables_dir"], "baseline_benchmark.csv")
    df.to_csv(out, index=False)
    LOGGER.info(f"Saved benchmark table to {out}")
    LOGGER.info("\n" + str(df))
    return df

if __name__ == "__main__":
    run()
