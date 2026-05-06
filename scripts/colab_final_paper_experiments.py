from pathlib import Path
import os
import sys
import zipfile
import shutil
import json
import yaml
import torch

# User settings
ZIP_PATH = "genomic-sequence-framework.zip"
EXTRACT_ROOT = "."

# Directory containing the required raw data files:
# hg19.pickle, ENCFF300IYQ.bed.gz, ATAC_data.bed.gz, ATAC_test_regions.bed.gz
DATA_SRC_DIR = "data"

# Directory where exported run outputs should be copied
RESULTS_EXPORT_DIR = "results_export"

def resolve_path(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())

def detect_project_dir(extract_root: str) -> str:
    root = Path(extract_root).resolve()
    candidates = []
    for child in root.iterdir():
        if child.is_dir() and (child / "requirements.txt").exists() and (child / "src").exists():
            candidates.append(child)
    if not candidates:
        if (root / "requirements.txt").exists() and (root / "src").exists():
            return str(root)
        raise FileNotFoundError("Could not detect extracted project directory.")
    candidates.sort()
    return str(candidates[0])

ZIP_PATH = resolve_path(ZIP_PATH)
EXTRACT_ROOT = resolve_path(EXTRACT_ROOT)
DATA_SRC_DIR = resolve_path(DATA_SRC_DIR)
RESULTS_EXPORT_DIR = resolve_path(RESULTS_EXPORT_DIR)

import copy
import pandas as pd

ZIP_PATH = "genomic-sequence-framework.zip"
EXTRACT_ROOT = "."
DATA_SRC_DIR = "data"
RESULTS_EXPORT_DIR = "genomic_results_final_paper"

SEEDS = [42, 43, 44]
TASKS = ["atac", "ctcf"]
ABLATIONS = {
    "full_model": {},
    "no_rc": {"model": {"rc_fusion": "none"}},
    "no_gru": {"model": {"rnn_layers": 0}},
    "no_dilation": {"model": {"dilations": [1, 1, 1, 1, 1, 1]}},
}

ZIP_PATH = resolve_path(ZIP_PATH)
EXTRACT_ROOT = resolve_path(EXTRACT_ROOT)
DATA_SRC_DIR = resolve_path(DATA_SRC_DIR)
RESULTS_EXPORT_DIR = resolve_path(RESULTS_EXPORT_DIR)

if ZIP_PATH.endswith(".zip") and os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_ROOT)
    PROJECT_DIR = detect_project_dir(EXTRACT_ROOT)
else:
    PROJECT_DIR = detect_project_dir(EXTRACT_ROOT)

print("Project directory:", PROJECT_DIR)

os.chdir(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

os.system("pip install -q -r requirements.txt")

os.makedirs("data/raw", exist_ok=True)
required_files = [
    "hg19.pickle",
    "ENCFF300IYQ.bed.gz",
    "ATAC_data.bed.gz",
    "ATAC_test_regions.bed.gz",
]
for fname in required_files:
    src = os.path.join(DATA_SRC_DIR, fname)
    dst = os.path.join("data/raw", fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("Copied:", fname)
    else:
        print("Missing source file:", src)

config_path = "configs/base_config.yaml"
with open(config_path, "r") as f:
    base_cfg = yaml.safe_load(f)

base_cfg["paths"]["genome_path"] = "data/raw/hg19.pickle"
base_cfg["paths"]["ctcf_bed_path"] = "data/raw/ENCFF300IYQ.bed.gz"
base_cfg["paths"]["atac_bed_path"] = "data/raw/ATAC_data.bed.gz"
base_cfg["paths"]["atac_test_bed_path"] = "data/raw/ATAC_test_regions.bed.gz"
base_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

from src.experiments.run_final_best_model import run as run_model

all_results = []
for task in TASKS:
    print("Running task:", task)
    for ablation_name, ablation_changes in ABLATIONS.items():
        print("Ablation:", ablation_name)
        for seed in SEEDS:
            print("Seed:", seed)
            cfg = copy.deepcopy(base_cfg)
            cfg["task"] = task
            cfg["seed"] = seed
            for section, changes in ablation_changes.items():
                if section not in cfg:
                    cfg[section] = {}
                cfg[section].update(changes)

            tmp_config_path = f"configs/tmp_{task}_{ablation_name}_{seed}.yaml"
            with open(tmp_config_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            out = run_model(config_path=tmp_config_path)
            metrics = out["val_metrics"]
            all_results.append({
                "task": task,
                "ablation": ablation_name,
                "seed": seed,
                **metrics,
            })

df = pd.DataFrame(all_results)
os.makedirs("results/final_paper", exist_ok=True)
df.to_csv("results/final_paper/all_runs.csv", index=False)

summary = df.groupby(["task", "ablation"]).agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    mean_auroc=("auroc", "mean"),
    std_auroc=("auroc", "std"),
    mean_auprc=("auprc", "mean"),
    std_auprc=("auprc", "std"),
).reset_index()
summary.to_csv("results/final_paper/summary.csv", index=False)

print("Summary")
print(summary)

os.makedirs(RESULTS_EXPORT_DIR, exist_ok=True)
dst = os.path.join(RESULTS_EXPORT_DIR, "final_paper_results")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree("results/final_paper", dst)

print("Saved to:", dst)
print("Done")
