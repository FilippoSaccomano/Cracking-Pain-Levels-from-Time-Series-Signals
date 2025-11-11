# Pirate Pain Level Classification

## 🎯 Project Overview

This project implements a deep learning solution for classifying pain levels in pirates based on multivariate time-series sensor data. The goal is to predict one of three pain categories:

- **no_pain**: No pain detected
- **low_pain**: Low level of pain  
- **high_pain**: High level of pain

## 📊 Dataset Description

### Input Features

The dataset contains time-series measurements captured from pirate subjects with the following features:

| Feature Category | Description | Count |
|-----------------|-------------|-------|
| **Pain Surveys** | Self-reported pain indicators | 4 features (pain_survey_1 to pain_survey_4) |
| **Body Characteristics** | Physical attributes (n_legs, n_hands, n_eyes) | 3 features |
| **Joint Angles** | Angular measurements from body joints | 31 features (joint_00 to joint_30) |

### Data Structure

- **Training Set**: 661 samples with labels
- **Test Set**: Unlabeled samples for prediction
- **Sequence Length**: 160 time steps per sample
- **Total Features**: 38 features per time step (excluding sample_index and time)

### Target Variable

Three-class categorical variable representing pain levels with class imbalance:
- Distribution varies across classes (handled via SMOTE resampling)

## 🔧 Installation & Setup

### Requirements

```bash
# Python 3.8+
pip install numpy pandas matplotlib seaborn
pip install tensorflow scikit-learn imbalanced-learn
pip install joblib tqdm jupyter
```

### Environment Setup

You can also use the provided `environment.yml` file if you prefer conda:

```bash
conda env create -f environment.yml
conda activate challenge_an2dl
```

## 🚀 Pipeline Architecture

### 1. Data Loading and Exploration (Sections 2.x)

**Purpose**: Load training data and understand its structure

**Steps**:
- Load CSV files containing time-series features and labels
- Analyze data dimensions, feature types, and temporal patterns
- Visualize class distribution to identify imbalance
- Explore time-series patterns for different feature categories

**Key Insights**:
- Data shows temporal dependencies requiring sequence modeling
- Class imbalance requires special handling
- Multiple feature types (categorical and continuous) need different preprocessing

### 2. Data Preprocessing (Sections 3.x)

**Purpose**: Transform raw data into model-ready format

**Steps**:

a) **Feature Preprocessing** (Section 3.1)
   - Convert categorical variables (zero/one/two/three) to numeric (0/1/2/3)
   - Handle missing values using mode imputation
   - Separate identifier columns (sample_index, time) from features

b) **Sequence Creation** (Section 3.2)
   - Group time steps by sample_index
   - Sort by time to maintain temporal order
   - Create 3D sequences: (num_samples, time_steps, num_features)

c) **Feature Scaling** (Section 3.3)
   - Apply StandardScaler to normalize all features
   - Fit on training data, transform both train and test
   - Improves model convergence and performance

d) **Sequence Padding** (Section 3.4)
   - Pad sequences to uniform length (max_length=160)
   - Zero-padding for shorter sequences
   - Enable batch processing in neural networks

e) **Label Encoding** (Section 3.5)
   - Convert text labels to integers: no_pain→0, low_pain→1, high_pain→2
   - Required for neural network classification

f) **Train-Validation Split** (Section 3.6)
   - 80% training, 20% validation split
   - Stratified split to maintain class proportions
   - Random seed (42) for reproducibility

### 3. Model Architecture (Sections 4.x)

**Purpose**: Define neural network architectures and training components

**Components**:

a) **Custom F1-Score Metric** (Section 4.1)
   - Implements macro-averaged F1-score for Keras
   - Primary evaluation metric for multi-class imbalanced classification
   - Tracks per-class precision and recall during training

b) **Model Architectures** (Section 4.2)
   Three different architectures tested:
   
   - **LSTM Model**:
     - Stacked LSTM layers (128 → 64 units)
     - Captures long-term temporal dependencies
     - Masking layer to handle padded sequences
     
   - **GRU Model**:
     - Stacked GRU layers (128 → 64 units)
     - Similar to LSTM but computationally lighter
     - Good alternative for temporal modeling
     
   - **CNN-LSTM Hybrid**:
     - Conv1D layer (64 filters) for local pattern extraction
     - MaxPooling for dimensionality reduction
     - LSTM layer (64 units) for temporal dependencies
     - Combines spatial and temporal feature learning

   **Common Architecture Elements**:
   - Masking layer (handles zero-padded sequences)
   - Dropout layers (0.3 rate) for regularization
   - Dense layers with ReLU activation
   - Softmax output layer (3 classes)

c) **Training Utilities** (Section 4.3)
   - **Early Stopping**: Prevents overfitting, monitors val_f1_score
   - **Learning Rate Reduction**: Adapts learning rate when plateauing
   - **SMOTE Resampling**: Balances classes by generating synthetic samples

### 4. Model Training (Sections 5.x)

**Purpose**: Train models and select best configuration

**Strategy**:

a) **Ensemble Grid Search** (Section 5.1)
   - Train multiple model types with different hyperparameters
   - Parallel training for efficiency (joblib)
   - Parameter grid includes:
     - Model types: LSTM, GRU, CNN_LSTM
     - Learning rates: 7e-4 to 1e-3
     - Dropout: 0.3
     - SMOTE: k_neighbors=5
   - 50 epochs with early stopping
   - Batch size: 32

b) **Ensemble Predictions** (Section 5.2)
   - Combine predictions from all trained models
   - Weighted voting based on individual model F1-scores
   - More weight to better-performing models

c) **Best Model Selection** (Section 5.3)
   - Compare individual models and ensemble
   - Select best single model based on validation F1-score
   - Display performance comparison table

d) **Final Model Training** (Section 5.4)
   - Retrain best configuration with more epochs (80)
   - Extended training for better convergence
   - Final model used for test predictions

e) **Training Visualization** (Section 5.5)
   - Plot loss, accuracy, and F1-score curves
   - Verify no overfitting (train vs validation)
   - Confirm proper convergence

### 5. Model Evaluation (Sections 6.x)

**Purpose**: Assess model performance on validation data

**Metrics**:

a) **Classification Report** (Section 6.1)
   - Precision, recall, F1-score per class
   - Macro-averaged F1-score (primary metric)
   - Weighted F1-score
   - Overall accuracy

b) **Confusion Matrix** (Section 6.2)
   - Visualize prediction patterns
   - Identify misclassification trends
   - Per-class accuracy breakdown

**Evaluation Focus**:
- F1-score prioritized over accuracy due to class imbalance
- Macro averaging treats all classes equally
- Confusion matrix reveals which classes are confused

### 6. Test Predictions (Sections 7.x)

**Purpose**: Generate predictions for submission

**Steps**:

a) **Load Test Data** (Section 7.1)
   - Read test CSV file
   - Verify data structure matches training data

b) **Preprocess Test Data** (Section 7.2)
   - Apply same preprocessing pipeline as training
   - Use training scaler (no refitting)
   - Create padded sequences with same dimensions

c) **Generate Predictions** (Section 7.3)
   - Use final trained model
   - Convert probabilities to class labels
   - Visualize prediction distribution

d) **Create Submission** (Section 7.4)
   - Format: sample_index (zero-padded), label
   - Save as CSV file
   - Ready for competition submission

## 📈 Results

### Model Performance

The ensemble approach combining LSTM, GRU, and CNN-LSTM architectures achieves strong performance on the validation set:

- **Primary Metric**: Macro F1-Score
- **Evaluation**: Stratified validation set (20% of training data)
- **Handling Imbalance**: SMOTE oversampling technique

### Key Performance Factors

1. **Data Augmentation**: SMOTE balances minority classes
2. **Ensemble Learning**: Multiple architectures capture different patterns
3. **Custom Metric**: F1-score optimizes for balanced class performance
4. **Regularization**: Dropout and early stopping prevent overfitting
5. **Feature Engineering**: Proper scaling and sequence padding

## 📁 File Structure

```
challenge_an2dl/
├── README.md                          # This file
├── pain_pirate_analysis_cleaned.ipynb # Main notebook (cleaned version)
├── pain_pirate_analysis.ipynb        # Original notebook
├── pirate_pain_train.csv             # Training features
├── pirate_pain_train_labels.csv      # Training labels
├── pirate_pain_test.csv              # Test features
├── submission.csv                     # Generated predictions
├── environment.yml                    # Conda environment file
└── prova.py                           # Legacy code file
```

## 🎮 How to Run

### Option 1: Run Complete Pipeline

Open and execute `pain_pirate_analysis_cleaned.ipynb` in Jupyter:

```bash
jupyter notebook pain_pirate_analysis_cleaned.ipynb
```

Execute cells sequentially from top to bottom. The notebook will:
1. Load and explore data
2. Preprocess features
3. Train multiple models
4. Generate predictions
5. Create `submission.csv`

### Option 2: Run Specific Sections

The notebook is modular - you can run specific sections:
- **Sections 1-3**: Data exploration and preprocessing only
- **Sections 4-5**: Model training only (requires preprocessed data)
- **Sections 6-7**: Evaluation and prediction only (requires trained model)

## 🧪 Experimentation Tips

### Improving Model Performance

1. **Hyperparameter Tuning**:
   - Adjust LSTM/GRU units (try 256, 128)
   - Modify dropout rates (0.2-0.5)
   - Experiment with learning rates (1e-4 to 1e-3)

2. **Data Augmentation**:
   - Try different SMOTE parameters (k_neighbors)
   - Experiment with other oversampling techniques
   - Add time-series augmentation (jittering, scaling)

3. **Architecture Modifications**:
   - Add more LSTM layers
   - Increase dense layer sizes
   - Try bidirectional LSTM
   - Add attention mechanisms

4. **Training Strategy**:
   - Increase epochs (with early stopping)
   - Adjust batch size (16, 64)
   - Try different optimizers (SGD, AdamW)
   - Implement learning rate schedules

5. **Ensemble Improvements**:
   - Train more diverse models
   - Use stacking instead of voting
   - Optimize ensemble weights

## 🔍 Troubleshooting

### Common Issues

**Issue**: Out of memory error during training
- **Solution**: Reduce batch size or model complexity

**Issue**: Model not learning (flat loss curve)
- **Solution**: Check learning rate, verify data preprocessing

**Issue**: Overfitting (large gap between train and validation)
- **Solution**: Increase dropout, add more regularization, reduce model complexity

**Issue**: Poor minority class performance
- **Solution**: Adjust SMOTE parameters, try class weights in loss function

## 📚 Technical Details

### Why LSTM/GRU?

Time-series data has temporal dependencies. Recurrent networks:
- Maintain hidden states across time steps
- Learn long-term patterns
- Handle variable-length sequences (with masking)

### Why SMOTE?

Class imbalance can bias models toward majority class. SMOTE:
- Generates synthetic minority samples
- Balances training set
- Improves minority class recall

### Why Ensemble?

Different architectures capture different patterns:
- LSTM: Long-term dependencies
- GRU: Efficient temporal modeling
- CNN-LSTM: Local patterns + temporal context
- Ensemble combines strengths of each

### Why F1-Score?

With class imbalance:
- Accuracy can be misleading
- F1-score balances precision and recall
- Macro averaging treats all classes equally

## 🤝 Contributing

This is an academic project. For questions or improvements:
1. Review the code and documentation
2. Test modifications on validation set first
3. Document changes clearly

## 📄 License

Academic project for AN2DL course.

## 🙏 Acknowledgments

- Course: Artificial Neural Networks and Deep Learning
- Dataset: Pirate Pain Classification Challenge
- Framework: TensorFlow/Keras
- Libraries: scikit-learn, imbalanced-learn, pandas, numpy

---

**Note**: This notebook prioritizes code clarity and educational value. All steps are documented and explained to facilitate learning and reproducibility.
