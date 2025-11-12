# Pain Classification Pipeline - Multi-class LSTM

This repository contains a clean, production-ready pipeline for multi-class pain classification using LSTM on time series data.

## Overview

The pipeline processes pirate pain dataset with time-series joint motion data, survey responses, and static body features to classify pain levels using a deep LSTM neural network.

## Files Structure

- `pain_classification_pipeline.ipynb` - Main pipeline notebook
- `pirate_pain_train.csv` - Training data (gitignored)
- `pirate_pain_train_labels.csv` - Training labels (gitignored)
- `pirate_pain_test.csv` - Test data (gitignored)
- `sample_submission.csv` - Submission format template (gitignored)
- `best_config.json` - Best model configuration (generated)
- `submission.csv` - Final predictions (generated)
- `submission_probs.csv` - Prediction probabilities (generated)
- `figures/` - Visualization outputs (generated)

## Pipeline Features

### 1. Data Processing
- Automatic data loading and label merging
- Time-based sequence sorting
- Comprehensive statistics and sequence length analysis

### 2. Feature Engineering
- **JOINT features**: 31 continuous joint motion features (joint_00 to joint_30)
- **SURVEY features**: 4 ordinal pain survey features (pain_survey_1 to pain_survey_4)
- **BODY features**: Static subject characteristics (categorical and continuous)

### 3. Preprocessing
- Linear interpolation for missing JOINT values (NaN only, not zeros)
- Categorical BODY feature mapping (zero/one/two/three → 0/1/2/3)
- Median imputation for missing BODY categorical values
- StandardScaler for JOINT and continuous BODY features
- MinMaxScaler for SURVEY features to preserve ordinality
- Static BODY features replicated across all time steps

### 4. Sequence Length Uniformization
- T_target automatically determined from training data (max sequence length)
- Edge padding for shorter sequences (repeat last frame)
- Truncation for longer sequences
- Same T_target applied to validation and test sets

### 5. Model Architecture
- Input: (T_target, num_features)
- SpatialDropout1D (0.15)
- LSTM Layer 1: configurable units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1
- LayerNormalization
- LSTM Layer 2: configurable units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1
- LayerNormalization
- Global Average + Max Pooling (concatenated)
- Dense head: configurable units, ReLU activation
- Dropout (0.3)
- Output: Dense(num_classes, softmax)
- Optimizer: AdamW with configurable learning rate, weight decay, clipnorm=1.0

### 6. Hyperparameter Grid Search
- 5 configurations tested
- Varies: LSTM units (96/128/160), head units (64/96), learning rate (5e-4/6e-4/7e-4), weight decay (1e-4/2e-4)
- Selection based on macro F1 score on validation set
- Best configuration saved to `best_config.json`

### 7. Training
- 80/20 train/validation split (stratified)
- EarlyStopping: patience=10, restore_best_weights=True
- ReduceLROnPlateau: factor=0.5, patience=4, min_lr=1e-5
- Batch size: 32
- Max epochs: 100

### 8. Evaluation Metrics
- Accuracy and F1 score (macro and per-class)
- Confusion matrix
- Classification report
- ROC curves (one-vs-rest)
- Precision-Recall curves (one-vs-rest)
- Probability distribution histograms per class

### 9. Visualizations
All saved to `./figures/`:
- `training_curves.png` - Loss and accuracy curves
- `confusion_matrix.png` - Validation confusion matrix
- `roc_pr_curves.png` - ROC and PR curves
- `probability_distributions.png` - Predicted probability distributions
- `feature_distributions.png` - Scaled feature distributions

### 10. Test Predictions
- Predictions on test set using best model
- `submission.csv` - Class predictions matching sample_submission format
- `submission_probs.csv` - Probability scores for all classes

## Usage

1. Place your CSV data files in the repository root:
   - `pirate_pain_train.csv`
   - `pirate_pain_train_labels.csv`
   - `pirate_pain_test.csv`
   - `sample_submission.csv`

2. Run the notebook cells sequentially:
   ```bash
   jupyter notebook pain_classification_pipeline.ipynb
   ```

3. The pipeline will:
   - Load and preprocess data
   - Determine optimal sequence length (T_target)
   - Run grid search to find best hyperparameters
   - Train final model with best configuration
   - Generate evaluation plots
   - Create submission files

## Requirements

See `environment.yml` for complete dependencies. Key packages:
- Python 3.10
- TensorFlow 2.16
- Keras 3.12
- scikit-learn 1.7
- pandas 2.3
- numpy 1.26
- matplotlib 3.10
- seaborn 0.13

## Key Design Decisions

1. **No windowing**: Full sequences processed as-is
2. **No data augmentation**: Clean baseline approach
3. **T_target from training only**: Test set not used for any parameter selection
4. **Edge padding**: Preserves temporal information better than zero padding
5. **Separate scaling for feature groups**: Preserves feature characteristics
6. **F1 macro for model selection**: Handles class imbalance
7. **AdamW optimizer**: Better generalization with weight decay

## Output Summary

After running, the notebook prints:
- T_target value
- Best configuration parameters
- Validation macro F1 and accuracy
- Per-class F1 scores
- Paths to all generated files

## Notes

- Fixed random seed (42) for reproducibility
- All data files (*.csv) are gitignored per .gitignore
- Model checkpoints not saved (can be added if needed)
- Public leaderboard not used for model/parameter selection
