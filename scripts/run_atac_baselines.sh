#!/usr/bin/env bash
set -euo pipefail
python scripts/run_experiment.py --task atac --experiment baselines "$@"
