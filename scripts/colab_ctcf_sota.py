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
RUN_SOTA_COMPARE = True

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

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

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
        print("Copied:", fname)
    else:
        print("Missing source file:", src)

config_path = "configs/base_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["task"] = "ctcf"
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

print("Running task:", cfg["task"])

if RUN_BASELINES:
    print("Running CTCF baselines")
    from src.experiments.run_baselines import run as run_baselines
    print(run_baselines(config_path=config_path))

if RUN_FINAL_MODEL:
    print("Running CTCF final model")
    from src.experiments.run_final_best_model import run as run_final_model
    final_out = run_final_model(config_path=config_path)
    print(final_out["val_metrics"])

if RUN_SOTA_COMPARE:
    print("Running CTCF SOTA comparison")
    from src.experiments.run_sota_compare import run as run_sota
    print(run_sota(config_path=config_path))

print("Result files")
for root, _, files in os.walk("results"):
    for f in files:
        print(os.path.join(root, f))

os.makedirs(RESULTS_EXPORT_DIR, exist_ok=True)
dst = os.path.join(RESULTS_EXPORT_DIR, "results")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree("results", dst)
shutil.copy2("configs/base_config.yaml", os.path.join(RESULTS_EXPORT_DIR, "ctcf_config_used.yaml"))

print("Saved outputs to:", RESULTS_EXPORT_DIR)
print("Done")
