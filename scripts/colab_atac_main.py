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

RUN_BASELINES = True
RUN_FINAL_MODEL = True
RUN_CONTEXT_SWEEP = False
RUN_KERNEL_SWEEP = False
RUN_DEPTH_SWEEP = False
RUN_DROPOUT_SWEEP = False
RUN_OPTIMIZER_SWEEP = False
RUN_RESIDUAL_COMPARE = False
RUN_ATTENTION_COMPARE = False
RUN_INTERPRETABILITY = False

if ZIP_PATH.endswith(".zip") and os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_ROOT)
    PROJECT_DIR = detect_project_dir(EXTRACT_ROOT)
else:
    PROJECT_DIR = detect_project_dir(EXTRACT_ROOT)

print("Project directory:", PROJECT_DIR)
print("Top-level files:", os.listdir(PROJECT_DIR))

os.chdir(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

os.system("pip install -q -r requirements.txt")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU is not enabled in this runtime.")

os.makedirs("data/raw", exist_ok=True)
for sub in ["checkpoints", "figures", "tables", "logs", "predictions"]:
    os.makedirs(f"results/{sub}", exist_ok=True)

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
        print(f"Copied: {fname}")
    else:
        print(f"Missing source file: {src}")

print("data/raw contents:", os.listdir("data/raw"))

config_path = "configs/base_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["paths"]["genome_path"] = "data/raw/hg19.pickle"
cfg["paths"]["ctcf_bed_path"] = "data/raw/ENCFF300IYQ.bed.gz"
cfg["paths"]["atac_bed_path"] = "data/raw/ATAC_data.bed.gz"
cfg["paths"]["atac_test_bed_path"] = "data/raw/ATAC_test_regions.bed.gz"
cfg["paths"]["checkpoints_dir"] = "results/checkpoints"
cfg["paths"]["figures_dir"] = "results/figures"
cfg["paths"]["tables_dir"] = "results/tables"
cfg["paths"]["logs_dir"] = "results/logs"
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

with open(config_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

from src.utils.config_utils import load_yaml
cfg = load_yaml(config_path)
print("Loaded task:", cfg["task"])

if RUN_BASELINES:
    print("Running baselines")
    from src.experiments.run_baselines import run as run_baselines
    baseline_df = run_baselines(config_path=config_path)
    print(baseline_df)

if RUN_FINAL_MODEL:
    print("Running final model")
    from src.experiments.run_final_best_model import run as run_final_best_model
    final_out = run_final_best_model(config_path=config_path)
    print(final_out["val_metrics"])

if RUN_CONTEXT_SWEEP:
    from src.experiments.run_context_sweep import run as run_context_sweep
    print(run_context_sweep(config_path=config_path))

if RUN_KERNEL_SWEEP:
    from src.experiments.run_kernel_sweep import run as run_kernel_sweep
    print(run_kernel_sweep(config_path=config_path))

if RUN_DEPTH_SWEEP:
    from src.experiments.run_depth_sweep import run as run_depth_sweep
    print(run_depth_sweep(config_path=config_path))

if RUN_DROPOUT_SWEEP:
    from src.experiments.run_dropout_sweep import run as run_dropout_sweep
    print(run_dropout_sweep(config_path=config_path))

if RUN_OPTIMIZER_SWEEP:
    from src.experiments.run_optimizer_sweep import run as run_optimizer_sweep
    print(run_optimizer_sweep(config_path=config_path))

if RUN_RESIDUAL_COMPARE:
    from src.experiments.run_residual_compare import run as run_residual_compare
    print(run_residual_compare(config_path=config_path))

if RUN_ATTENTION_COMPARE:
    from src.experiments.run_attention_compare import run as run_attention_compare
    print(run_attention_compare(config_path=config_path))

if RUN_INTERPRETABILITY:
    from src.experiments.run_interpretability_suite import run as run_interpretability_suite
    print(run_interpretability_suite(config_path=config_path))

print("Result files")
for root, dirs, files in os.walk("results"):
    for f in files:
        print(os.path.join(root, f))

os.makedirs(RESULTS_EXPORT_DIR, exist_ok=True)
dst_results = os.path.join(RESULTS_EXPORT_DIR, "results")
if os.path.exists(dst_results):
    shutil.rmtree(dst_results)
shutil.copytree("results", dst_results)
shutil.copy2("configs/base_config.yaml", os.path.join(RESULTS_EXPORT_DIR, "base_config_used.yaml"))

print("Saved outputs to:", RESULTS_EXPORT_DIR)
print("Done")
