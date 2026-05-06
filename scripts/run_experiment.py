#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    p = argparse.ArgumentParser(description="Run a configured genomic experiment.")
    p.add_argument("--task", choices=["atac", "ctcf"], required=True)
    p.add_argument("--experiment", choices=["baselines", "final", "sota"], required=True)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base_config.yaml"))
    p.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    return p.parse_args()

def patch_config(path: Path, task: str, device: str | None, seed: int | None) -> Path:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["task"] = task
    if device is not None:
        cfg["device"] = device
    if seed is not None:
        cfg["seed"] = seed
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return Path(tmp.name)

def main():
    args = parse_args()
    cfg_path = patch_config(Path(args.config), args.task, args.device, args.seed)

    if args.experiment == "baselines":
        from src.experiments.run_baselines import run
    elif args.experiment == "final":
        from src.experiments.run_final_best_model import run
    else:
        from src.experiments.run_sota_compare import run

    out = run(config_path=str(cfg_path))
    print(out)

if __name__ == "__main__":
    main()
