# Genomic Sequence Framework

A modular deep learning framework for **regulatory DNA prediction** from sequence, with experiments on:

- **ATAC-seq accessibility prediction**
- **CTCF binding prediction**

This repository contains the full pipeline for:
- loading genomic intervals from BED files
- extracting DNA windows from **hg19**
- one-hot sequence encoding
- robust positive/negative sample generation
- training baseline and advanced neural models
- controlled comparison against **SOTA-inspired** architectures
- exporting tables, figures, logs, and predictions

The main proposed model combines **reverse-complement-aware fusion**, **dilated residual CNN blocks**, a **bidirectional GRU**, and a **gated aggregation head**.

---

## Overview

The project studies binary regulatory prediction from DNA sequence. Given a fixed-length genomic window, the model predicts whether the region corresponds to a regulatory signal of interest.

Two tasks are included:

- **ATAC**: open chromatin / accessibility prediction
- **CTCF**: sequence-based prediction of CTCF-associated regions

The repository is organized so that:

- `src/` contains the core implementation
- `scripts/` contains launchers and workflow scripts
- `configs/` contains YAML experiment settings
- `results/` stores generated outputs
- `runs/` can store archived final experiment results

---

## Main Contributions

- **Modular genomic pipeline** for loading, preprocessing, training, and evaluation
- **Reverse-complement-aware dilated recurrent architecture** for regulatory DNA prediction
- **Controlled evaluation** on ATAC and CTCF with baseline and SOTA-inspired comparisons

---

## System Architecture

Add the system architecture image below.

<p align="center">
  <img src="docs/figures/system_architecture.png" alt="System architecture" width="900">
</p>

<p align="center">
  <em><strong>Figure 1.</strong> Proposed reverse-complement-aware model for genomic regulatory prediction.</em>
</p>

### Architecture summary
The flagship model follows this sequence:

1. DNA window extraction  
2. One-hot encoding  
3. Reverse-complement-aware processing  
4. Dilated residual 1D convolutions  
5. Bidirectional GRU sequence modeling  
6. Gated feature aggregation  
7. Binary prediction head  

This design aims to capture both local motifs and broader contextual dependencies while respecting DNA strand symmetry.

---

## Experimental Summary Figure



<p align="center">
  <img width="1680" height="1230" alt="Image" src="https://github.com/user-attachments/assets/9f8c4782-fa93-483d-a2be-17e7724f1be5" />
</p>

<p align="center">
  <em><strong>Figure 2.</strong> Multi-panel summary of experimental comparisons, metrics, and training behavior across ATAC and CTCF.</em>
</p>

---

## Repository Structure

```text
genomic-sequence-framework/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
├── data/
├── docs/
│   └── figures/
├── reports/
├── results/
├── runs/
├── scripts/
├── src/
└── tests/
```

### Important folders
- **`src/`**: core code
- **`scripts/`**: launchers and Colab workflows
- **`configs/`**: experiment configuration files
- **`data/`**: raw and processed data directories
- **`results/`**: generated outputs during runs
- **`runs/`**: final archived experiment outputs
- **`docs/figures/`**: figures used in README, paper, or slides

---

## Tasks and Data

The repository expects the following raw files in `data/raw/`:

```text
data/raw/hg19.pickle
data/raw/ENCFF300IYQ.bed.gz
data/raw/ATAC_data.bed.gz
data/raw/ATAC_test_regions.bed.gz
```

### Data pipeline
The main data workflow is:

1. load BED intervals  
2. define positive regions  
3. generate structured negatives  
4. extract fixed-length sequence windows  
5. encode sequences as one-hot tensors  
6. train and evaluate models  

A major feature of the framework is the **improved negative sampling strategy**, which reduces trivial shortcuts and makes the classification setup more realistic.

---

## Models Included

### Baselines
- Logistic baseline
- MLP baseline
- 1D CNN baseline

### SOTA-inspired models
- DeepSEA-style
- DanQ-style
- Basenji-style

### Proposed model
- RC-Dilated-BiGRU-Gated

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/genomic-sequence-framework.git
cd genomic-sequence-framework
```

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

---

## Quick Start

### ATAC
```bash
python scripts/run_experiment.py --task atac --experiment baselines
python scripts/run_experiment.py --task atac --experiment final
python scripts/run_experiment.py --task atac --experiment sota
```

### CTCF
```bash
python scripts/run_experiment.py --task ctcf --experiment baselines
python scripts/run_experiment.py --task ctcf --experiment final
python scripts/run_experiment.py --task ctcf --experiment sota
```

### Shell wrappers
```bash
bash scripts/run_atac_baselines.sh
bash scripts/run_atac_final.sh
bash scripts/run_atac_sota.sh
bash scripts/run_ctcf_baselines.sh
bash scripts/run_ctcf_final.sh
bash scripts/run_ctcf_sota.sh
```

---

## Colab Workflows

The repository also includes Colab-oriented workflow scripts under `scripts/`:

- `colab_one_cell.py`
- `colab_atac_main.py`
- `colab_atac_sota.py`
- `colab_ctcf_sota.py`
- `colab_final_paper_experiments.py`

These handle:
- project extraction
- dependency installation
- data copying
- config patching
- experiment execution
- output export

---

## Configuration

The main config files are:

- `configs/base_config.yaml`
- `configs/atac.yaml`
- `configs/ctcf.yaml`

These define:
- task
- paths
- dataset parameters
- model hyperparameters
- training settings

---

## Outputs

Generated experiment outputs are written to `results/`:

- `results/checkpoints/`
- `results/figures/`
- `results/logs/`
- `results/predictions/`
- `results/tables/`

For final GitHub archiving, you can copy selected outputs into `runs/`, for example:

```text
runs/
├── atac_main/
├── atac_sota/
├── ctcf_sota/
└── final_paper/
```

Recommended files to archive:
- CSV tables
- JSON metrics
- PNG figures
- config files
- training history logs

Usually checkpoints are excluded from version control.

---

## Extending the Framework

You can extend the repository by:

- adding new BED-based tasks
- adding new models under `src/models/`
- registering new architectures in the model factory
- creating new experiment runners in `src/experiments/`
- exposing new workflows through `scripts/`

---

## Citation

If you use this repository, please cite the accompanying paper or report.

```bibtex
@misc{genomicsequenceframework,
  title        = {Genomic Sequence Framework},
  author       = {Your Name},
  year         = {2025},
  note         = {Project repository}
}
```

---

## References

The project is informed by prior work including:

- Zhou & Troyanskaya, **DeepSEA**, *Nature Methods* (2015)
- Quang & Xie, **DanQ**, *Nucleic Acids Research* (2016)
- Kelley et al., **Basenji**, *Genome Research* (2018)
- Buenrostro et al., **ATAC-seq**, *Nature Methods* (2013)
- Ong & Corces, **CTCF**, *Nature Reviews Genetics* (2014)
- Loshchilov & Hutter, **AdamW**, *ICLR* (2019)
- Zhou et al., reverse-complement equivariance in genomics, *PMLR* (2022)

---

## Final Note

This repository is designed to be a clean, modular, and practical framework for regulatory genomics experiments. For the best entry into the codebase, start with:

1. `configs/base_config.yaml`
2. `scripts/run_experiment.py`
3. `src/`
4. `results/` and `runs/`
