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

from src.models.rc_dilated_bigru_gated import RCDilatedCNNBiGRUGated
from src.training.train_wrapper import train_model_boosted_iter
LOGGER = get_logger("final_best_model")

class BedPeaksDatasetTest(torch.utils.data.IterableDataset):
    def __init__(self, peaks_df, genome, context_length):
        super().__init__()
        self.peaks_df = peaks_df
        self.genome = genome
        self.context_length = context_length
    def __iter__(self):
        from src.data.sequence_encoder import one_hot
        for row in self.peaks_df.itertuples():
            midpoint = int(0.5 * (row.start + row.end))
            seq = self.genome[row.chrom][midpoint - self.context_length // 2: midpoint + self.context_length // 2]
            if len(seq) == self.context_length:
                yield torch.tensor(one_hot(seq), dtype=torch.float32)

def predict_unlabeled(model, loader, device):
    model.eval()
    preds = []
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for xb in loader:
            xb = xb.to(device, non_blocking=True).float()
            preds.append(torch.sigmoid(model(xb).squeeze()).detach().cpu().numpy())
    import numpy as np
    return np.concatenate(preds)

def build_model(config):
    m = config["model"]
    L = config["data"]["context_length"]
    return RCDilatedCNNBiGRUGated(
        seq_len=L, stem_channels=m.get("stem_channels",64), block_channels=m.get("block_channels",128),
        n_blocks=m.get("n_blocks",6), kernel_size=m.get("kernel_size",7),
        dilations=tuple(m.get("dilations",[1,2,4,8,16,32])),
        rnn_hidden=m.get("rnn_hidden",128), rnn_layers=m.get("rnn_layers",1),
        p_drop=m.get("dropout",0.15), n_hidden=m.get("hidden_dim",256),
        n_output_channels=m.get("n_outputs",1), rc_fusion=m.get("rc_fusion","lse"),
        lse_temp=m.get("lse_temp",2.0)
    )

def run(config_path="configs/base_config.yaml"):
    config = load_yaml(config_path)
    set_seed(config.get("seed",42))
    make_dirs(config)
    LOGGER.info("Loading genome...")
    genome = load_genome(config["paths"]["genome_path"])
    LOGGER.info("Loading task data...")
    train_df, val_df, _ = load_task_data(config, test=True)

    unlabeled_test_df = None
    if config.get("task","atac").lower() == "atac" and os.path.exists(config["paths"].get("atac_test_bed_path","")):
        unlabeled_test_df = pd.read_csv(config["paths"]["atac_test_bed_path"], sep="\t", compression="infer", names=["chrom","start","end"])

    train_loader = build_loader(train_df, genome, config, improved=True)
    val_loader = build_loader(val_df, genome, config, improved=False)
    device = torch.device(config.get("device","cuda") if torch.cuda.is_available() else "cpu")

    LOGGER.info("Building flagship model...")
    model = build_model(config).to(device)
    LOGGER.info("Training...")
    model, train_accs, val_accs = train_model_boosted_iter(
        model=model, train_loader=train_loader, val_loader=val_loader,
        epochs=config["train"].get("epochs",20), steps_per_epoch=config["train"].get("steps_per_epoch",300),
        patience=config["train"].get("patience",6), base_lr=config["train"].get("lr",3e-3),
        verbose=True, device=device,
    )

    probs, labels = evaluate_model(model, val_loader, device)
    threshold = select_best_threshold(labels, probs)
    metrics = compute_classification_metrics(labels, probs, threshold=threshold)
    metrics["best_threshold"] = float(threshold)

    with open(os.path.join(config["paths"]["tables_dir"], "final_best_model_val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    ckpt = os.path.join(config["paths"]["checkpoints_dir"], "final_best_model.pt")
    torch.save(model.state_dict(), ckpt)
    with open(os.path.join(config["paths"]["logs_dir"], "final_best_model_history.json"), "w") as f:
        json.dump({"train_accs":[float(x) for x in train_accs], "val_accs":[float(x) for x in val_accs]}, f, indent=2)

    if unlabeled_test_df is not None:
        pred_loader = DataLoader(
            BedPeaksDatasetTest(unlabeled_test_df, genome, config["data"]["context_length"]),
            batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"]
        )
        test_probs = predict_unlabeled(model, pred_loader, device)
        out_df = unlabeled_test_df.copy().reset_index(drop=True)
        out_df["probability"] = test_probs[:len(out_df)]
        out_df.to_csv("results/predictions/final_best_model_predictions.csv", index=False)

    LOGGER.info("Done.")
    return {"model": model, "val_metrics": metrics, "checkpoint": ckpt}

if __name__ == "__main__":
    run()
