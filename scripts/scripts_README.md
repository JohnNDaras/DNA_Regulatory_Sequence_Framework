# `scripts/` Directory Reference

This document describes the full `scripts/` directory of the **genomic-sequence-framework** repository. The `scripts/` folder contains the user-facing entry points that sit on top of the `src/` package. While `src/` contains the project’s core logic, the `scripts/` directory is responsible for making that logic easy to run in practice.

The guiding principle of this folder is simplicity:

- `src/` implements the underlying data/model/training/evaluation logic
- `scripts/` provides convenient ways to launch experiments without editing internal Python modules

This folder therefore contains two categories of files:

1. **portable launchers** that can be used locally or from a terminal
2. **Colab-oriented workflow scripts** that bundle environment setup, data copying, config patching, experiment execution, and results export into a single file

Because this is the final repository version, the scripts are written to be:

- repository-agnostic
- relatively environment-agnostic
- easy to edit through a clearly marked user settings block
- explicit rather than overly abstract

---

# 1. Overview of the `scripts/` folder

```text
scripts/
├── run_experiment.py
├── run_atac_baselines.sh
├── run_atac_final.sh
├── run_atac_sota.sh
├── run_ctcf_baselines.sh
├── run_ctcf_final.sh
├── run_ctcf_sota.sh
├── colab_one_cell.py
├── colab_atac_main.py
├── colab_atac_sota.py
├── colab_ctcf_sota.py
└── colab_final_paper_experiments.py
```

At a high level:

- `run_experiment.py` is the main generic launcher
- `run_*.sh` files are tiny convenience wrappers around `run_experiment.py`
- `colab_*.py` files are full workflows intended to be pasted into a Colab cell or adapted into notebook cells

---

# 2. General design philosophy

The scripts are intentionally split this way because the project has two very different usage modes:

## 2.1 Local / terminal execution
A user cloning the repository locally usually wants something like:

```bash
python scripts/run_experiment.py --task atac --experiment baselines
```

or the even shorter wrappers:

```bash
bash scripts/run_atac_baselines.sh
```

This usage mode assumes:
- Python environment is already active
- dependencies are installed
- the repository already exists as a local folder
- data files have already been placed into `data/raw/`

## 2.2 Colab execution
A Colab user often wants a single script that:
- mounts Drive
- unzips the project
- installs requirements
- copies raw data into the correct location
- patches config paths
- runs experiments
- copies results back out

This is much more procedural, which is why the Colab scripts are longer.

---

# 3. `scripts/run_experiment.py`

## Purpose
This is the primary portable Python launcher for the repository.

It is the cleanest non-Colab entry point because it delegates almost everything to the `src/experiments/` runners.

## Main responsibilities
- parse command-line arguments
- load a base config path
- override selected config values (task, device, seed)
- write a temporary patched config
- dispatch to the correct experiment runner

## Expected usage
Examples:

```bash
python scripts/run_experiment.py --task atac --experiment baselines
python scripts/run_experiment.py --task atac --experiment final
python scripts/run_experiment.py --task atac --experiment sota

python scripts/run_experiment.py --task ctcf --experiment baselines
python scripts/run_experiment.py --task ctcf --experiment final
python scripts/run_experiment.py --task ctcf --experiment sota
```

## Command-line arguments

### `--task`
Allowed values:
- `atac`
- `ctcf`

This determines which task-specific split logic and dataset path logic the downstream experiment code will use.

### `--experiment`
Allowed values:
- `baselines`
- `final`
- `sota`

This determines which experiment module is called:
- `src.experiments.run_baselines`
- `src.experiments.run_final_best_model`
- `src.experiments.run_sota_compare`

### `--config`
Optional path to a YAML config file.

Default:
- `configs/base_config.yaml`

Use this if you want to run an alternative configuration file while preserving the same launcher logic.

### `--device`
Optional override for computation device, for example:
- `cpu`
- `cuda`

This is useful when forcing CPU execution or when debugging on machines without GPU.

### `--seed`
Optional integer seed override.

This is useful for:
- reproducibility
- multi-seed runs
- quick stability checks

## Internal flow

### 3.1 Determine repository root
The script computes:

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
```

This makes the launcher portable as long as it remains inside the repository under `scripts/`.

### 3.2 Parse arguments
The script uses `argparse` to validate the task and experiment type.

### 3.3 Patch the config
The helper function `patch_config(...)`:
- loads the original YAML config
- updates `task`
- optionally updates `device`
- optionally updates `seed`
- writes a temporary YAML file
- returns the temporary path

This design avoids mutating the original config permanently.

### 3.4 Dispatch to experiment runner
Based on `--experiment`, the script imports the correct runner and executes it.

This is intentionally explicit instead of using dynamic import magic.

## Why this script matters
If someone wants a single recommended non-Colab way to run the project, this is it.

It is:
- minimal
- clear
- aligned with the repository structure
- much easier to maintain than duplicating complex shell logic

---

# 4. Shell wrappers

The shell wrappers are intentionally tiny. They do not contain experiment logic themselves. Instead, they are shortcuts for the most common runs.

---

## 4.1 `scripts/run_atac_baselines.sh`

### Purpose
Convenience wrapper for ATAC baseline runs.

### Command it executes
```bash
python scripts/run_experiment.py --task atac --experiment baselines "$@"
```

### Why it exists
This is useful for users who prefer short commands or want to place these wrappers into job submission scripts.

---

## 4.2 `scripts/run_atac_final.sh`

### Purpose
Convenience wrapper for the ATAC flagship model run.

### Command it executes
```bash
python scripts/run_experiment.py --task atac --experiment final "$@"
```

### Typical use
Run the final ATAC model without remembering the full Python command.

---

## 4.3 `scripts/run_atac_sota.sh`

### Purpose
Convenience wrapper for ATAC SOTA-inspired comparison runs.

### Command it executes
```bash
python scripts/run_experiment.py --task atac --experiment sota "$@"
```

### Why it matters
This gives a quick command for producing the ATAC comparison table and figure.

---

## 4.4 `scripts/run_ctcf_baselines.sh`

### Purpose
Convenience wrapper for CTCF baseline runs.

### Command it executes
```bash
python scripts/run_experiment.py --task ctcf --experiment baselines "$@"
```

---

## 4.5 `scripts/run_ctcf_final.sh`

### Purpose
Convenience wrapper for the CTCF flagship model run.

### Command it executes
```bash
python scripts/run_experiment.py --task ctcf --experiment final "$@"
```

---

## 4.6 `scripts/run_ctcf_sota.sh`

### Purpose
Convenience wrapper for CTCF SOTA-inspired comparison runs.

### Command it executes
```bash
python scripts/run_experiment.py --task ctcf --experiment sota "$@"
```

---

# 5. Colab workflow scripts

The Colab scripts are full procedural workflows rather than minimal launchers. Their job is to move from “I have a zip and some data files” to “I have experiment outputs saved somewhere.”

They all share the same general structure:

1. define user-editable settings
2. extract the project zip if needed
3. auto-detect the repository root
4. install requirements
5. prepare `data/raw/` and `results/`
6. copy raw data files into the repo
7. patch `configs/base_config.yaml`
8. run one or more experiments
9. copy outputs to an export folder

Each workflow exists because different project phases had slightly different needs.

---

## 5.1 Shared path pattern in Colab scripts

The final Colab scripts use a simple editable block such as:

```python
ZIP_PATH = "genomic-sequence-framework.zip"
EXTRACT_ROOT = "."
DATA_SRC_DIR = "data"
RESULTS_EXPORT_DIR = "results_export"
```

This means the scripts do **not** assume:
- a specific Google Drive account
- a specific `/content/drive/MyDrive/...` path
- a personal directory like `Assignment 2`
- a weird extracted folder name

They instead rely on:
- an explicit zip path
- an explicit extract root
- an explicit data directory
- an explicit results export directory

That makes them easier to adapt to:
- Colab
- local Jupyter
- containerized environments
- shared servers

---

## 5.2 Project auto-detection helper

The Colab scripts define:

```python
def detect_project_dir(extract_root: str) -> str:
    ...
```

### Purpose
Automatically finds the extracted project directory by scanning for a folder containing:
- `requirements.txt`
- `src/`

### Why it matters
This avoids hardcoding something like:
- `/content/genomic-sequence-framework`
- `/content/genomic-sequence-framework-final-polished`

and makes the script more portable.

---

# 6. `scripts/colab_one_cell.py`

## Purpose
Single-file workflow for the main ATAC run.

## Role in the repository
This file is effectively the “default Colab workflow” for the project.

In the final repository version it is aligned with the ATAC main workflow, which includes:
- baselines
- final model
- optional extra experiments if enabled

## Main user settings
- `ZIP_PATH`
- `EXTRACT_ROOT`
- `DATA_SRC_DIR`
- `RESULTS_EXPORT_DIR`
- experiment toggles such as:
  - `RUN_BASELINES`
  - `RUN_FINAL_MODEL`
  - `RUN_CONTEXT_SWEEP`
  - `RUN_KERNEL_SWEEP`
  - `RUN_DEPTH_SWEEP`
  - `RUN_DROPOUT_SWEEP`
  - `RUN_OPTIMIZER_SWEEP`
  - `RUN_RESIDUAL_COMPARE`
  - `RUN_ATTENTION_COMPARE`
  - `RUN_INTERPRETABILITY`

## Why it exists separately from `run_experiment.py`
`run_experiment.py` assumes the environment is already prepared.
`colab_one_cell.py` prepares the environment itself.

## Internal steps

### 6.1 Extraction
If `ZIP_PATH` exists and ends with `.zip`, the script extracts it.

### 6.2 Project detection
The script auto-detects the repository root.

### 6.3 Dependency installation
Runs:
```python
os.system("pip install -q -r requirements.txt")
```

### 6.4 Device check
Reports whether CUDA is available and prints GPU name if present.

### 6.5 Directory preparation
Ensures:
- `data/raw`
- `results/checkpoints`
- `results/figures`
- `results/tables`
- `results/logs`
- `results/predictions`

exist.

### 6.6 Data copying
Copies the required files into `data/raw/`:
- `hg19.pickle`
- `ENCFF300IYQ.bed.gz`
- `ATAC_data.bed.gz`
- `ATAC_test_regions.bed.gz`

### 6.7 Config patching
Updates:
- genome path
- BED paths
- result output paths
- device

in `configs/base_config.yaml`

### 6.8 Experiment execution
Conditionally runs:
- baselines
- final model
- optional sweeps and comparisons

### 6.9 Results export
Copies `results/` and the patched config into `RESULTS_EXPORT_DIR`.

## When to use it
Use this file when you want the closest thing to a full one-cell notebook workflow for the original ATAC project runs.

---

# 7. `scripts/colab_atac_main.py`

## Purpose
Dedicated Colab workflow for the original ATAC main experiments.

## Relationship to `colab_one_cell.py`
In the final repository, this file is essentially the same core workflow as `colab_one_cell.py`, but it exists separately for naming clarity.

## Included runs
- ATAC baselines
- ATAC final model
- optional sweeps and comparisons

## Why keep both files
Some users like a single generic “one cell” file; others prefer a file whose name tells them exactly what it does. Keeping both makes the repository easier to navigate.

## When to use it
Use this when you want:
- the ATAC baselines
- the ATAC final model
- optional ATAC auxiliary experiments

without switching to a different task.

---

# 8. `scripts/colab_atac_sota.py`

## Purpose
Colab workflow for ATAC experiments including SOTA-inspired comparison.

## Main use case
This is the script to use when reproducing the ATAC comparison among:
- baselines
- final flagship model
- SOTA-inspired models

## Main toggles
- `RUN_BASELINES`
- `RUN_FINAL_MODEL`
- `RUN_SOTA_COMPARE`

## Internal flow
This script follows the same preparation logic as the main ATAC Colab script, but the experiment section is simpler and more focused.

### 8.1 Setup
- extract project
- auto-detect repo
- install requirements
- prepare data/results directories
- copy raw files
- patch config

### 8.2 Run sequence
1. baselines
2. final model
3. SOTA comparison

### 8.3 Output handling
Copies:
- `results/`
- `config_used.yaml`

into the export directory.

## Why it matters
This is the Colab workflow most directly tied to the comparison results used in the architecture study.

---

# 9. `scripts/colab_ctcf_sota.py`

## Purpose
Colab workflow for CTCF experiments including SOTA-inspired comparison.

## Main use case
This is the CTCF counterpart of `colab_atac_sota.py`.

## Key difference from the ATAC version
It explicitly patches:

```python
cfg["task"] = "ctcf"
```

so that downstream experiment logic uses the CTCF-specific data split and task path behavior.

## Included runs
- CTCF baselines
- CTCF final model
- CTCF SOTA-inspired comparison

## Internal flow
Very similar to the ATAC SOTA script, with the key task switch being the main difference.

### 9.1 Setup
- extract project
- detect repo
- install requirements
- copy data
- patch config for CTCF

### 9.2 Run sequence
1. CTCF baselines
2. CTCF final model
3. CTCF SOTA comparison

### 9.3 Export
Copies:
- results directory
- the patched CTCF config

to `RESULTS_EXPORT_DIR`.

## Why it matters
This is the cleanest workflow for generating the CTCF comparison outputs used in the second major results block of the project.

---

# 10. `scripts/colab_final_paper_experiments.py`

## Purpose
Runs the multi-task, multi-seed, ablation-oriented workflow used for expanded final-paper experiments.

## Important note
This is the most advanced and most experimental script in the folder. It is included because it represents a real project workflow, but it is inherently more fragile than the baseline/final/SOTA scripts because it attempts to run multiple ablations over multiple seeds and tasks.

This is not a flaw in the script itself; it reflects the fact that ablation workflows are more likely to expose modeling assumptions or unsupported configuration combinations.

## Main settings
- `ZIP_PATH`
- `EXTRACT_ROOT`
- `DATA_SRC_DIR`
- `RESULTS_EXPORT_DIR`
- `SEEDS`
- `TASKS`
- `ABLATIONS`

## Default task list
- `atac`
- `ctcf`

## Default seed list
- `42`
- `43`
- `44`

## Default ablations
- `full_model`
- `no_rc`
- `no_gru`
- `no_dilation`

## Internal logic

### 10.1 Extract and setup
Same general environment setup as other Colab scripts.

### 10.2 Load and patch base config
The script uses the base config as a template and modifies it programmatically.

### 10.3 Nested loop
The core loop is:

- over tasks
- over ablations
- over seeds

For each combination, it:
- copies the base config
- writes task and seed
- applies ablation-specific modifications
- writes a temporary config file
- calls `src.experiments.run_final_best_model.run(...)`

### 10.4 Save aggregated results
The script writes:
- `results/final_paper/all_runs.csv`
- `results/final_paper/summary.csv`

The summary file stores grouped mean and standard deviation statistics.

### 10.5 Export results
Copies the final-paper results folder into the export directory.

## Why it matters
This script is the closest thing in the repository to a structured robustness/ablation pipeline.

## Why it should be used carefully
Because some ablations may require explicit support in the model definition, this script is best treated as:
- a useful final-paper experiment tool
- not the most “plug-and-play” script in the folder

---

# 11. How the scripts relate to the `src/` package

The scripts do not implement the core machine learning logic. They orchestrate the code from `src/`.

Typical dependency flow:

```text
scripts/*
  -> src.experiments.run_*
      -> src.data.*
      -> src.models.*
      -> src.training.*
      -> src.evaluation.*
      -> src.utils.*
```

This means:

- if you are debugging model behavior, look in `src/`
- if you are debugging path setup or environment setup, look in `scripts/`

---

# 12. Which script should a user choose?

A new user should choose based on the task they want to perform.

## 12.1 I want a local terminal run
Use:

```bash
python scripts/run_experiment.py ...
```

or the tiny shell wrappers.

## 12.2 I want the original ATAC baseline + final workflow in Colab
Use:
- `scripts/colab_atac_main.py`
or
- `scripts/colab_one_cell.py`

## 12.3 I want ATAC comparison including SOTA-inspired models
Use:
- `scripts/colab_atac_sota.py`

## 12.4 I want CTCF comparison including SOTA-inspired models
Use:
- `scripts/colab_ctcf_sota.py`

## 12.5 I want seeds + ablations + multi-task aggregation
Use:
- `scripts/colab_final_paper_experiments.py`

---

# 13. Practical strengths of the `scripts/` folder

The scripts directory is useful because it keeps the repository practical without overcomplicating the `src/` package.

Its main strengths are:

- clear experiment entry points
- separation of local vs Colab execution patterns
- explicit config patching
- easy results export
- readable, linear workflows

This is especially valuable in research projects, where reproducibility often depends as much on the orchestration logic as on the model code.

---

# 14. Limitations and intentional boundaries

The scripts are not intended to be a full workflow engine or job scheduler. They do not provide:

- advanced CLI experiment composition
- cluster orchestration
- distributed training launch logic
- automatic run versioning
- experiment database tracking

Those features are intentionally out of scope for this project.

Instead, the scripts are designed to be:
- simple enough to inspect manually
- explicit enough to edit quickly
- reliable enough for a compact research codebase

---

# 15. Recommended reading order for `scripts/`

If someone wants to understand the folder quickly, the best order is:

1. `run_experiment.py`
2. `colab_atac_main.py`
3. `colab_atac_sota.py`
4. `colab_ctcf_sota.py`
5. `colab_final_paper_experiments.py`
6. the shell wrappers

This order moves from:
- the simplest generic launcher
to
- progressively more specialized workflows

---

# 16. Summary

The `scripts/` directory is the execution layer of the repository.

It gives the project:
- a simple generic launcher (`run_experiment.py`)
- task-specific shell shortcuts
- complete Colab workflows for the main project stages

The most important files are:

- `scripts/run_experiment.py`
- `scripts/colab_atac_main.py`
- `scripts/colab_atac_sota.py`
- `scripts/colab_ctcf_sota.py`

Together, they make the repository usable without requiring users to directly manipulate internal modules in `src/`.

In short:

- `src/` contains the project logic
- `scripts/` makes that logic runnable