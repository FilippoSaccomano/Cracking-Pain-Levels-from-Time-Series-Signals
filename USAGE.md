# Usage Guide - Pain Classification Pipeline

## Quick Start

### Prerequisites
1. Ensure you have the conda environment set up:
   ```bash
   conda env create -f environment.yml
   conda activate AN2L
   ```

2. Place your data files in the repository root:
   - `pirate_pain_train.csv`
   - `pirate_pain_train_labels.csv`
   - `pirate_pain_test.csv`
   - `sample_submission.csv`

### Running the Pipeline

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook pain_classification_pipeline.ipynb
   ```

2. **Execute cells sequentially** (Kernel → Restart & Run All, or run each cell manually)

3. **Monitor progress**: The notebook will print status messages at each step

### Expected Runtime
- Data loading and preprocessing: < 1 minute
- Grid search (5 configurations): 30-60 minutes (depends on data size)
- Final training and evaluation: 10-20 minutes
- Total: ~1-2 hours

## Output Files

After completion, you'll find:

### Generated Files
- `best_config.json` - Best hyperparameter configuration
- `submission.csv` - Class predictions for test set
- `submission_probs.csv` - Probability scores for all classes
- `figures/` directory with:
  - `training_curves.png`
  - `confusion_matrix.png`
  - `roc_pr_curves.png`
  - `probability_distributions.png`
  - `feature_distributions.png`

### Console Output
The final cell prints a summary including:
- T_target (sequence length)
- Best configuration parameters
- Validation F1 macro and accuracy
- Per-class F1 scores
- Paths to output files

## Data Format Requirements

### Training Data (`pirate_pain_train.csv`)
Must contain:
- `sample_index` - Unique identifier for each sequence
- `time` - Time step within sequence
- `joint_00` to `joint_30` - Joint motion features (31 continuous features)
- `pain_survey_1` to `pain_survey_4` - Survey responses (ordinal: 0/1/2)
- Body features (e.g., `n_legs`, `n_hands`, `n_eyes`) - Static subject features

### Training Labels (`pirate_pain_train_labels.csv`)
Must contain:
- `sample_index` - Matches training data
- Label column (e.g., `pain_level`) - Target class (0, 1, 2, ...)

### Test Data (`pirate_pain_test.csv`)
Same structure as training data, but without label column

### Sample Submission (`sample_submission.csv`)
Template showing expected submission format:
- First column: `id` or `sample_index`
- Remaining columns: Prediction columns

## Customization

### Adjusting Grid Search
Edit the `grid_configs` list in Section 7:
```python
grid_configs = [
    {'units1': 128, 'units2': 64, 'head_units': 64, 'lr': 6e-4, 'wd': 1e-4},
    # Add more configurations...
]
```

### Modifying Model Architecture
Edit the `build_model` function in Section 6 to change:
- Dropout rates
- Number of LSTM layers
- Dense layer sizes
- Activation functions

### Changing Training Parameters
Modify in Section 7:
- `epochs` - Maximum training epochs (default: 100)
- `batch_size` - Batch size (default: 32)
- `test_size` - Validation split ratio (default: 0.2)

### Early Stopping Patience
Adjust in callback configuration:
```python
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  # Change this value
    restore_best_weights=True
)
```

## Troubleshooting

### Memory Issues
If you run out of memory:
1. Reduce `batch_size` (try 16 or 8)
2. Reduce model complexity (smaller LSTM units)
3. Use fewer grid search configurations

### Long Training Time
To speed up:
1. Reduce grid search configurations (test fewer)
2. Reduce `epochs` maximum
3. Increase early stopping patience (less likely to help)

### Data Loading Errors
Common issues:
- **File not found**: Ensure CSV files are in repository root
- **Column mismatch**: Verify column names match expected format
- **Encoding issues**: Try adding `encoding='utf-8'` to `pd.read_csv()`

### Model Not Converging
Try:
1. Adjust learning rate (lower: 1e-4, higher: 1e-3)
2. Change weight decay (try 0 or 1e-3)
3. Increase model capacity (more LSTM units)
4. Check for data quality issues

## Advanced Usage

### Running on GPU
The notebook automatically uses GPU if available. To verify:
```python
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

### Saving Model Weights
Add after training:
```python
best_model.save_weights('model_weights.h5')
```

Load later:
```python
model = build_model(...)
model.load_weights('model_weights.h5')
```

### Ensemble Predictions
Train multiple models with different seeds and average predictions:
```python
predictions = []
for seed in [42, 123, 456]:
    # Train model with seed
    # Get predictions
    predictions.append(model.predict(X_test))

ensemble_pred = np.mean(predictions, axis=0).argmax(axis=1)
```

### Cross-Validation
Instead of single train/val split:
```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
    # Train model on fold
    # Evaluate on validation
```

## Best Practices

1. **Always run the entire notebook** from top to bottom after any changes
2. **Check data statistics** in Section 2 to understand your dataset
3. **Monitor training curves** to detect overfitting
4. **Review confusion matrix** to identify problematic classes
5. **Don't use test set** for any model selection or parameter tuning
6. **Document any changes** to the pipeline for reproducibility

## Support

For issues or questions:
1. Check the README.md for detailed pipeline documentation
2. Review the problem statement in the repository
3. Check TensorFlow/Keras documentation for model-specific issues
4. Verify data format matches expectations

## Notes

- **Reproducibility**: Fixed seed ensures consistent results across runs
- **Validation Strategy**: Stratified split maintains class distribution
- **Metric Choice**: F1 macro handles class imbalance better than accuracy
- **Test Processing**: Same preprocessing applied to test as training
- **No Data Leakage**: All scalers fit only on training data
