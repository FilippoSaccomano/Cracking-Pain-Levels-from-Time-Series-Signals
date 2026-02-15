# Pain Pirate: Cracking Pain Levels from Time-Series Signals

This repository contains our AN2DL challenge work on **multivariate time-series classification** for the *Pirate Pain* dataset.  
Goal: classify each subject into one of three classes: **no pain**, **low pain**, or **high pain**.

The project compares multiple deep-learning architectures and shows that a **Conv1D + BiLSTM** model is the most effective and stable solution.

## Project highlights

- Dataset size (from report): **105,760 time steps**, **661 subjects**
- Input features: physiological/body-configuration and joint-movement time-series signals
- Main challenge: **strong class imbalance** (high-pain class is minority)
- Core strategy:
  - categorical encoding + time-feature enrichment
  - sliding-window sequence generation (**window size = 60**)
  - stratified train/validation split
  - class-weighted loss for imbalance handling

## Best-performing model

According to `Report.pdf`, the strongest model is:

- **Conv1D + BiLSTM** (Macro-F1: **0.9592** on validation in the comparison table)

Compared architectures in this repo:

- Conv1D + BiLSTM (`1_Best_Model_CNN_BiLSTM.ipynb`)
- Conv1D + BiLSTM + Attention (`2_CNN_BiLSTM_Attention.ipynb`)
- Conv1D + BiGRU + Attention (`3_CNN_BiGRU_Attention.ipynb`)

## Repository structure

- `1_Best_Model_CNN_BiLSTM.ipynb` → best baseline architecture
- `2_CNN_BiLSTM_Attention.ipynb` → attention variant with BiLSTM
- `3_CNN_BiGRU_Attention.ipynb` → attention variant with BiGRU
- `Report.pdf` → full methodology, experiments, and discussion
- `environment.yml` → reproducible environment specification

## Environment setup

```bash
conda env create -f environment.yml
conda activate AN2L
jupyter notebook
```

Then open and run any notebook from top to bottom.

## Notes

- The notebooks are exploratory/experimental pipelines and include preprocessing, model training, and evaluation in one place.
- `environment.yml` was exported from the original team environment (Apple Silicon) and includes `tensorflow-macos` / `tensorflow-metal`; the notebooks in this repository primarily use **PyTorch**.
