# `src/` Directory Reference

This document describes the full `src/` package of the **genomic-sequence-framework** repository. It is intended to serve as a detailed technical reference for anyone reading, modifying, reproducing, or extending the project. The codebase is organized around a simple but important idea: take genomic intervals from BED files, convert them into fixed-length one-hot encoded DNA windows, train deep learning or classical neural baselines on those windows, and evaluate the resulting models on regulatory sequence prediction tasks such as **ATAC-seq accessibility** and **CTCF binding**.

The `src/` tree is the core of the repository. It contains:

- package initialization
- data loading and dataset generation
- model definitions
- training utilities
- evaluation helpers
- experiment entry points
- configuration and utility helpers

At a high level, the execution flow is:

1. a YAML config is loaded  
2. the genome and BED files are read  
3. train/validation splits are created  
4. a dataset and dataloaders are built  
5. a model is instantiated  
6. training is run with the boosted wrapper  
7. predictions are evaluated with classification metrics  
8. experiment outputs are written to `results/`

---

# 1. Top-level module layout

```text
src/
├── __init__.py
├── main.py
├── data/
├── evaluation/
├── experiments/
├── models/
├── training/
└── utils/
```

Each subpackage has a focused role:

- `data/`: loading genome/BED inputs and converting peaks into training samples
- `evaluation/`: metrics, threshold selection, and benchmark table utilities
- `experiments/`: experiment entry points used by scripts
- `models/`: all model definitions and the model factory
- `training/`: training loop helpers and training wrapper
- `utils/`: configuration, logging, paths, plotting, and seeding

---

# 2. `src/__init__.py`

## Purpose
Marks `src/` as a Python package.

## Notes
This file is intentionally minimal. It exists so imports like:

```python
from src.experiments.run_baselines import run
```

work correctly.

---

# 3. `src/main.py`

## Purpose
Minimal command-line entry point for loading a configuration and setting the random seed.

## Main responsibility
- parses `--config`
- loads the YAML configuration
- sets the random seed
- prints which config was loaded

## When it is useful
This file is best understood as a lightweight sanity-check entry point rather than the main experiment launcher. The project’s real experiment execution is handled by the files in `src/experiments/` and by `scripts/run_experiment.py`.

## Key behavior
- imports `load_yaml` from `src.utils.config_utils`
- imports `set_seed` from `src.utils.seed`
- loads `configs/base_config.yaml` by default

---

# 4. `src/utils/`

The `utils/` package contains the small helper functions that keep the rest of the code cleaner.

---

## 4.1 `src/utils/__init__.py`

### Purpose
Marks the `utils/` directory as a package.

---

## 4.2 `src/utils/config_utils.py`

### Purpose
Loads YAML configuration files.

### Main function
- `load_yaml(path: str)`

### Behavior
- opens the file at `path`
- parses it using `yaml.safe_load`
- returns the resulting Python dictionary

### Why it matters
Most of the project is configuration-driven. Experiment files call `load_yaml(...)` to get:
- data paths
- batch size
- context length
- model hyperparameters
- training settings

This function is intentionally tiny and focused.

---

## 4.3 `src/utils/logger.py`

### Purpose
Creates a consistent project logger.

### Main function
- `get_logger(name="genomic")`

### Behavior
- creates a standard Python `logging` logger
- attaches a `StreamHandler`
- formats messages as:
  - `[INFO] ...`
  - `[WARNING] ...`
  - etc.

### Why it matters
The experiment runners use this logger to print concise status messages while training or saving outputs.

---

## 4.4 `src/utils/paths.py`

### Purpose
Defines common top-level paths relative to the repository root.

### Variables
- `ROOT`
- `DATA_DIR`
- `RESULTS_DIR`

### Behavior
`ROOT` is computed using the location of the file itself, so these paths remain portable as long as the repository structure is preserved.

### Why it matters
This file provides a shared understanding of where the project lives on disk.

---

## 4.5 `src/utils/plotting.py`

### Purpose
Contains simple plotting utilities for training curves and metric bar plots.

### Main functions
- `plot_history(history, save_path=None)`
- `plot_metric_bar(df, metric_col="val_auprc", label_col="model", title=None, save_path=None)`

### `plot_history`
Plots:
- training accuracy
- validation accuracy

This is useful for exported history JSON files.

### `plot_metric_bar`
Builds a simple bar plot from a results dataframe.

Common use:
- plotting AUPRC comparisons across models

### Why it matters
The SOTA comparison experiment uses this file to produce the summary bar plot saved in `results/figures/`.

---

## 4.6 `src/utils/seed.py`

### Purpose
Controls randomness for reproducibility.

### Main function
- `set_seed(seed: int = 42)`

### Behavior
Sets seeds for:
- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA

### Why it matters
This is essential for:
- reproducible experiments
- fair model comparisons
- multi-seed analysis

---

# 5. `src/data/`

This package handles the transition from raw genomic inputs to actual model-ready tensors.

---

## 5.1 `src/data/__init__.py`

### Purpose
Marks the `data/` directory as a package.

---

## 5.2 `src/data/genome_loader.py`

### Purpose
Loads the reference genome object from disk.

### Main function
- `load_genome(path: str)`

### Behavior
- opens a pickle file
- loads the genome dictionary with `pickle.load`

### Expected structure
The project assumes the genome file is a dictionary-like object indexed by chromosome name, for example:
- `genome["chr1"]`
- `genome["chr2"]`

Each chromosome entry should return a DNA string.

### Why it matters
All sequence extraction depends on this object.

---

## 5.3 `src/data/bed_loader.py`

### Purpose
Loads BED files containing genomic intervals.

### Main function
- `load_bed(path: str, include_chr_x_y: bool = False)`

### Behavior
- reads the first three BED columns:
  - `chrom`
  - `start`
  - `end`
- optionally excludes `chrX` and `chrY`
- sorts rows
- drops duplicates
- resets the dataframe index

### Why it matters
This is the standard entry point for both:
- ATAC peak files
- CTCF peak files

---

## 5.4 `src/data/sequence_encoder.py`

### Purpose
Converts DNA strings into one-hot encoded arrays.

### Main constants and functions
- `BASES = {"A": 0, "C": 1, "G": 2, "T": 3}`
- `one_hot(sequence: str) -> np.ndarray`

### Output format
The returned array has shape:

```text
(4, sequence_length)
```

where the rows correspond to:
- A
- C
- G
- T

### Why it matters
This is the core representation used by every model in the repository.

---

## 5.5 `src/data/negative_sampling.py`

### Purpose
Placeholder module indicating that negative sampling logic lives in `dataset.py`.

### Why it exists
It preserves package structure and makes it obvious that negative sampling is a named concept in the project, even though the implementation is consolidated elsewhere in this repository version.

---

## 5.6 `src/data/dataset.py`

### Purpose
Defines the dataset objects that generate positive and negative training examples.

This file is one of the most important in the repository.

### Main classes
- `BedPeaksDataset`
- `BedPeaksDatasetBetter`

---

### 5.6.1 `BedPeaksDataset`

#### Purpose
Simple iterable dataset for generating:
- positive windows from peaks
- naive negative windows from gaps between neighboring peaks

#### Behavior
For each row in the peaks dataframe:
1. compute the midpoint of the peak
2. extract a centered fixed-length sequence window
3. yield a positive example with label `1.0`

Additionally, if two adjacent peaks on the same chromosome leave a gap, the midpoint of the gap may be used as a negative example with label `0.0`.

#### Why it matters
This is the simpler, earlier dataset formulation used for validation and for non-improved negative settings.

---

### 5.6.2 `BedPeaksDatasetBetter`

#### Purpose
Improved iterable dataset with more realistic negative sampling.

#### Key ideas
This class creates negatives that are harder and more biologically realistic by enforcing:
- a minimum distance from positive peaks
- GC matching
- filtering of windows with too many `N` bases
- multiple negatives per positive window

#### Important parameters
- `context_length`
- `n_neg`
- `min_gap`
- `max_tries`
- `gc_tol`
- `max_N_frac`
- `rng_seed`
- `chroms_keep`

#### Internal behavior
This class:
- groups peaks by chromosome
- stores interval starts/ends
- checks whether candidate negative midpoints are sufficiently far from peaks
- computes GC content
- rejects poor candidate windows
- yields multiple negatives per positive

#### Why it matters
This dataset is a major part of the project’s experimental rigor. It reduces shortcut learning and makes the training problem more realistic.

---

## 5.7 `src/data/datamodule.py`

### Purpose
Builds PyTorch dataloaders from peak data and the genome.

### Main function
- `build_dataloaders(train_df, val_df, genome, context_length, batch_size=512, num_workers=0)`

### Behavior
Creates:
- a training dataloader
- a validation dataloader

using `BedPeaksDataset`.

### Why it matters
This is a convenience helper for cases where the simpler dataset construction is enough.

---

# 6. `src/models/`

This package defines all supported neural architectures and baselines.

---

## 6.1 `src/models/__init__.py`

### Purpose
Marks the `models/` directory as a package.

---

## 6.2 `src/models/base_model.py`

### Purpose
Defines a minimal base model class.

### Main class
- `BaseModel(nn.Module)`

### Why it matters
It gives all models a shared inheritance root and keeps the model namespace consistent.

---

## 6.3 `src/models/logistic_baseline.py`

### Purpose
Implements the simplest baseline: logistic regression over flattened one-hot sequence input.

### Main class
- `LogisticBaseline`

### Behavior
- flattens the `(4, L)` tensor into a vector of length `4L`
- applies a single linear layer
- outputs one logit

### Why it matters
This is the weakest baseline and helps establish a lower bound for sequence-based prediction.

---

## 6.4 `src/models/mlp_baseline.py`

### Purpose
Implements a simple multilayer perceptron baseline.

### Main class
- `MLPBaseline`

### Behavior
- flattens the sequence
- applies:
  - linear
  - ReLU
  - linear

### Why it matters
This tests whether a non-convolutional nonlinear model can learn useful signal directly from the flattened sequence.

---

## 6.5 `src/models/cnn1d.py`

### Purpose
Implements a straightforward 1D CNN baseline.

### Main class
- `CNN1D`

### Main design
- stacked Conv1D blocks
- batch normalization
- dropout
- ELU activation
- max pooling
- MLP head

### Why it matters
This is a strong conventional sequence baseline and often outperforms the logistic/MLP baselines by a large margin.

---

## 6.6 `src/models/residual_cnn.py`

### Purpose
Implements a residual CNN baseline.

### Main classes
- `ResidualBlock1D`
- `ResidualCNN`

### Main design
- convolutional stem
- residual blocks
- max pooling
- linear head

### Why it matters
This introduces residual connections, which help deeper convolutional feature extraction.

---

## 6.7 `src/models/cnn_attention.py`

### Purpose
Implements a CNN plus attention baseline.

### Main class
- `CNNAttention`

### Main design
- convolutional feature extraction
- multi-head self-attention
- adaptive max pooling
- MLP head

### Why it matters
This tests whether explicit attention over learned sequence features improves over purely convolutional baselines.

---

## 6.8 `src/models/deepsea_model.py`

### Purpose
Implements a lightweight DeepSEA-inspired convolutional model.

### Main class
- `DeepSEAStyle`

### Main design
- three convolutional blocks
- batch normalization
- ReLU
- max pooling
- dropout
- global max-style summarization
- MLP classifier

### Why it matters
This provides a controlled comparison against an architecture family inspired by a foundational genomics model.

---

## 6.9 `src/models/danq_model.py`

### Purpose
Implements a lightweight DanQ-inspired model.

### Main class
- `DanQStyle`

### Main design
- convolutional feature extractor
- bidirectional LSTM
- dropout
- max pooling across sequence positions
- MLP head

### Why it matters
This is a classical hybrid CNN + recurrent sequence model and is a strong comparison point for your architecture.

---

## 6.10 `src/models/basenji_style_model.py`

### Purpose
Implements a lightweight Basenji/Basenji2-inspired dilated CNN.

### Main classes
- `BasenjiDilatedBlock`
- `BasenjiStyle`

### Main design
- convolutional stem
- repeated residual dilated convolution blocks
- adaptive average pooling
- MLP head

### Why it matters
This serves as the main dilated convolution comparison family.

---

## 6.11 `src/models/rc_dilated_bigru_gated.py`

### Purpose
Defines the project’s main architecture:
**reverse-complement aware dilated CNN + BiGRU + gated refinement + pooled classification head**

This is the flagship model of the repository.

### Main helper functions and classes
- `reverse_complement_batch`
- `SpatialDropout1D`
- `SE1D`
- `DilatedResBlock`
- `AttnPool1D`
- `RCDilatedCNNBiGRUGated`

---

### 6.11.1 `reverse_complement_batch`
Takes a one-hot batch and:
- reverses channel order from A/C/G/T to T/G/C/A
- flips along sequence length

This creates the reverse complement representation.

---

### 6.11.2 `SpatialDropout1D`
Applies channel-wise dropout across the temporal axis by adapting `Dropout2d`.

---

### 6.11.3 `SE1D`
A squeeze-and-excitation block for channel reweighting.

---

### 6.11.4 `DilatedResBlock`
A residual block using:
- two dilated convolutions
- batch normalization
- ELU
- spatial dropout
- squeeze-and-excitation
- residual connection

This is the backbone unit used in the main model.

---

### 6.11.5 `AttnPool1D`
Computes learned attention scores across positions and returns an attention-weighted summary vector.

---

### 6.11.6 `RCDilatedCNNBiGRUGated`
This is the full model.

#### Main stages
1. **Stem**
   - Conv1D + BN + ELU + spatial dropout

2. **Projection**
   - 1x1 convolution to block channels

3. **Dilated residual stack**
   - multiple `DilatedResBlock`s with increasing dilations

4. **Reverse-complement feature fusion**
   - max / mean / log-sum-exp fusion
   - `lse` is the main mode used in the project

5. **Bidirectional GRU**
   - models sequential dependencies over the fused feature map

6. **Normalization and dropout**
   - applied to recurrent outputs

7. **Gated attention refinement**
   - builds a similarity matrix
   - applies a sigmoid gate
   - uses gated refinement plus residual behavior

8. **Three pooled summaries**
   - attention pooling
   - average pooling
   - max pooling

9. **Prediction head**
   - concatenates the pooled summaries
   - MLP head produces final logit

#### Why it matters
This file contains the model that defines the paper/project’s central technical contribution.

---

## 6.12 `src/models/model_factory.py`

### Purpose
Centralized model builder.

### Main function
- `build_model(model_cfg: dict, input_length: int)`

### Behavior
Chooses a model implementation from `model_cfg["name"]`.

Supported names include:
- `logistic_baseline`
- `mlp_baseline`
- `cnn1d`
- `residual_cnn`
- `cnn_attention`
- `deepsea_style`
- `danq_style`
- `basenji_style`
- `rc_dilated_bigru_gated`

### Why it matters
The experiment code stays cleaner because model selection is delegated to this file.

---

# 7. `src/training/`

This package defines how optimization is run.

---

## 7.1 `src/training/__init__.py`

### Purpose
Marks the `training/` directory as a package.

---

## 7.2 `src/training/checkpointing.py`

### Purpose
Saves model weights to disk.

### Main function
- `save_checkpoint(model, path: str)`

### Behavior
- creates the parent directory if needed
- saves `model.state_dict()` using `torch.save`

### Why it matters
This is the simplest checkpointing utility used in the project.

---

## 7.3 `src/training/early_stopping.py`

### Purpose
Provides an early stopping helper class.

### Main class
- `EarlyStopping`

### Behavior
Tracks:
- best validation value
- patience counter
- stop condition

### Note
The main boosted training wrapper uses a more embedded stopping logic, so this file functions more as a standalone utility than as the central stopping implementation.

---

## 7.4 `src/training/optimizer_factory.py`

### Purpose
Placeholder for optimizer construction.

### Current status
The file explicitly notes that the optimizer is built inside the boosted wrapper for this project version.

### Why it matters
This documents a design decision: the optimization stack was consolidated into the main training wrapper.

---

## 7.5 `src/training/run_one_epoch.py`

### Purpose
Placeholder for a more modular per-epoch runner.

### Current status
This file is not used in the current project version.

### Why it matters
It reflects a possible future refactor target, but not an active code path.

---

## 7.6 `src/training/train_wrapper.py`

### Purpose
Contains the main practical training implementation used by experiments.

### Main function
- `train_model_boosted_iter(...)`

### This is one of the most important files in training.

#### Major features
- device placement
- mixed precision training with AMP
- automatic class-imbalance handling with `pos_weight`
- `AdamW`
- `OneCycleLR`
- EMA-style moving average of model parameters
- validation tracking
- early stopping on validation accuracy
- returning best model state plus training history

#### Training behavior
- estimates class imbalance from batches
- constructs `BCEWithLogitsLoss` with `pos_weight`
- runs a fixed number of steps per epoch
- uses smoothed labels
- tracks training and validation accuracy
- stores best state based on validation accuracy
- stops after patience is exceeded

#### Why it matters
This wrapper is the central optimization engine behind:
- baselines
- final model
- SOTA comparisons

---

## 7.7 `src/training/trainer.py`

### Purpose
Higher-level trainer utilities used by experiment code.

### Main contents
This file includes helper logic for:
- evaluating models
- writing outputs
- orchestrating training/evaluation in experiment-friendly form

### Why it matters
It complements the lower-level training wrapper and experiment runners.

---

# 8. `src/evaluation/`

This package contains evaluation metrics and result formatting helpers.

---

## 8.1 `src/evaluation/__init__.py`

### Purpose
Marks the evaluation package.

---

## 8.2 `src/evaluation/metrics.py`

### Purpose
Computes the core classification metrics.

### Main function
- `compute_classification_metrics(y_true, y_prob, threshold=0.5)`

### Metrics returned
- accuracy
- F1
- AUROC
- AUPRC

### Behavior
- converts probabilities to hard predictions using `threshold`
- computes the metrics with scikit-learn
- gracefully handles failures in AUROC/AUPRC by returning `nan`

### Why it matters
This is the metric function used across the project.

---

## 8.3 `src/evaluation/thresholding.py`

### Purpose
Selects the best classification threshold.

### Main function
- `select_best_threshold(y_true, y_prob)`

### Behavior
- sweeps thresholds from `0.05` to `0.95`
- computes F1 at each threshold
- returns the threshold with best F1

### Why it matters
The project does not blindly use 0.5 for every model. It selects a validation-based threshold.

---

## 8.4 `src/evaluation/benchmark_table.py`

### Purpose
Builds a results dataframe and sorts it.

### Main function
- `build_benchmark_table(records)`

### Behavior
Creates a pandas dataframe and sorts by `val_auprc`.

### Why it matters
Useful for comparing multiple models consistently.

---

## 8.5 `src/evaluation/evaluator.py`

### Purpose
Placeholder noting that evaluation helpers are embedded directly in the experiment runners for this repository version.

### Why it matters
Documents that this responsibility was not split into a dedicated evaluator module in the current code.

---

## 8.6 `src/evaluation/split_robustness.py`

### Purpose
Provides an alternate CTCF chromosome split.

### Main function
- `alternate_ctcf_split(df)`

### Why it matters
This can be useful for robustness analysis across different genomic split definitions.

---

# 9. `src/experiments/`

This package contains the three most important experiment entry points in the repository.

---

## 9.1 `src/experiments/__init__.py`

### Purpose
Marks the experiments directory as a package.

---

## 9.2 `src/experiments/run_baselines.py`

### Purpose
Runs the baseline benchmark on the selected task.

### Main responsibilities
- load config
- set seed
- create output directories
- load genome
- load task-specific train/validation data
- build loaders
- instantiate baseline models
- train each model
- evaluate each model
- save metrics, checkpoints, benchmark CSV, and histories

### Task handling
This file supports:
- ATAC
- CTCF

The split logic is task-dependent:
- ATAC: validation on chr3/chr4
- CTCF: training on chr4–22, validation on chr2/chr3, optional test on chr1

### Why it matters
This is the main script used to compare standard baselines against the proposed model family.

---

## 9.3 `src/experiments/run_final_best_model.py`

### Purpose
Trains and evaluates the flagship model on the selected task.

### Main responsibilities
- load config and seed
- create output directories
- load genome
- load task data
- build improved-negative training loader
- build validation loader
- instantiate the flagship model
- train with `train_model_boosted_iter`
- compute validation probabilities
- choose the best threshold
- compute final metrics
- save:
  - validation metrics JSON
  - checkpoint
  - history JSON
  - unlabeled ATAC predictions if test BED is present

### Why it matters
This is the central script for generating the paper’s final single-model results.

---

## 9.4 `src/experiments/run_sota_compare.py`

### Purpose
Runs a controlled comparison among:
- baselines
- SOTA-inspired architectures
- the proposed model

### Main responsibilities
- load genome and task data
- construct the full model list
- train/evaluate each model
- save per-model metrics and histories
- save comparison CSV
- save AUPRC comparison plot

### Included families
- baseline
- DeepSEA-inspired
- DanQ-inspired
- Basenji-inspired
- ours

### Why it matters
This is the script that produces the main comparison table and figure for the architecture study.

---

# 10. How the `src/` code fits together

A typical experiment uses `src/` in the following order:

1. **config loaded**  
   `src/utils/config_utils.py`

2. **seed fixed**  
   `src/utils/seed.py`

3. **genome + BED loaded**  
   `src/data/genome_loader.py`  
   `src/data/bed_loader.py`

4. **dataset created**  
   `src/data/dataset.py`

5. **dataloader built**  
   via PyTorch `DataLoader`

6. **model built**  
   `src/models/model_factory.py`

7. **training run**  
   `src/training/train_wrapper.py`

8. **validation predictions collected**  
   experiment-local evaluation helpers

9. **metrics computed**  
   `src/evaluation/metrics.py`

10. **best threshold selected**  
    `src/evaluation/thresholding.py`

11. **CSV / JSON / figures written**  
    experiment runners + `src/utils/plotting.py`

---

# 11. Most important files to understand first

If someone is new to the repository, the best reading order is:

1. `src/experiments/run_final_best_model.py`  
2. `src/models/rc_dilated_bigru_gated.py`  
3. `src/data/dataset.py`  
4. `src/training/train_wrapper.py`  
5. `src/experiments/run_sota_compare.py`  
6. `src/evaluation/metrics.py`  
7. `src/models/model_factory.py`

That order gives the fastest understanding of:
- what the project does
- how the flagship model works
- how the data is formed
- how optimization is done
- how results are compared

---

# 12. Summary

The `src/` directory is the actual engine of the repository. It contains a compact but complete pipeline for regulatory DNA sequence modeling:

- **data ingestion**
- **sequence encoding**
- **negative sampling**
- **baseline and advanced models**
- **training**
- **evaluation**
- **experiment orchestration**

The design is intentionally pragmatic:
- simple YAML-driven configuration
- explicit experiment files
- PyTorch-based training
- easy export of tables, histories, and figures

The central technical identity of the project is defined by three files:

- `src/data/dataset.py`
- `src/training/train_wrapper.py`
- `src/models/rc_dilated_bigru_gated.py`

Together, they implement the project’s main claims:
- realistic sequence classification from BED/genome inputs
- controlled baseline and SOTA-inspired comparison
- a reverse-complement aware dilated CNN + BiGRU + gated-attention architecture for regulatory prediction