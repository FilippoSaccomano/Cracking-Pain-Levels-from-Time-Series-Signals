# %%
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Preprocessing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils import resample, class_weight
from imblearn.over_sampling import SMOTE

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("\n⚓ All libraries loaded successfully!")

# %%
# Load training data
X_train = pd.read_csv('pirate_pain_train.csv')
y_train = pd.read_csv('pirate_pain_train_labels.csv')

print("🏴‍☠️ Training Data Overview")
print("=" * 50)
print(f"Training features shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"\nNumber of unique samples: {X_train['sample_index'].nunique()}")
print(f"Time steps per sample (min-max): {X_train.groupby('sample_index').size().min()} - {X_train.groupby('sample_index').size().max()}")
print(f"\nFeature columns: {X_train.shape[1]}")

# %%
# Display first few rows
print("\n📊 First few rows of training data:")
display(X_train.head())

print("\n🏷️ First few labels:")
display(y_train.head())

# %%
# Label distribution
label_counts = y_train['label'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
axes[0].bar(label_counts.index, label_counts.values, color=['green', 'orange', 'red'])
axes[0].set_xlabel('Pain Level', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Pain Levels', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Pie chart
colors = {'no_pain': 'green', 'low_pain': 'orange', 'high_pain': 'red'}
pie_colors = [colors.get(label, 'gray') for label in label_counts.index]
axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=pie_colors)
axes[1].set_title('Pain Level Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📈 Label Statistics:")
print(label_counts)
print(f"\nClass balance: {label_counts.min() / label_counts.max():.2%}")

# %%
# Explore feature types
print("\n🔍 Feature Analysis:")
print("=" * 50)

# Identify feature groups
pain_survey_cols = [col for col in X_train.columns if 'pain_survey' in col]
body_char_cols = ['n_legs', 'n_hands', 'n_eyes']
joint_cols = [col for col in X_train.columns if 'joint_' in col]

print(f"\n📋 Pain Survey Features: {len(pain_survey_cols)}")
print(pain_survey_cols)

print(f"\n🧍 Body Characteristics: {len(body_char_cols)}")
print(body_char_cols)

print(f"\n🦴 Joint Measurements: {len(joint_cols)}")
print(f"Joints: joint_00 to joint_{len(joint_cols)-1:02d}")

print(f"\n✅ Total features (excluding sample_index, time): {len(pain_survey_cols) + len(body_char_cols) + len(joint_cols)}")

# %%
# Visualize a sample time series
sample_idx = 0
sample_data = X_train[X_train['sample_index'] == sample_idx].sort_values('time')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Time Series Visualization for Sample {sample_idx}', fontsize=16, fontweight='bold')

# Pain surveys
for col in pain_survey_cols:
    axes[0, 0].plot(sample_data['time'], sample_data[col], marker='o', label=col)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_title('Pain Survey Readings')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Body characteristics
for col in body_char_cols:
    if col in sample_data.columns:
        axes[0, 1].plot(sample_data['time'], sample_data[col], marker='s', label=col)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_title('Body Characteristics')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# First 5 joints
for i, col in enumerate(joint_cols[:5]):
    axes[1, 0].plot(sample_data['time'], sample_data[col], label=col, alpha=0.7)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Angle')
axes[1, 0].set_title('Joint Angles (0-4)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Last 5 joints
for col in joint_cols[-5:]:
    axes[1, 1].plot(sample_data['time'], sample_data[col], label=col, alpha=0.7)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Angle')
axes[1, 1].set_title(f'Joint Angles ({len(joint_cols)-5}-{len(joint_cols)-1})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
def preprocess_features(df):
    """
    Preprocess features: handle categorical variables and missing values
    """
    # Exclude identifier columns
    exclude_cols = ['sample_index', 'time']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Categorical columns available in the dataset
    categorical_cols = [col for col in ['n_legs', 'n_hands', 'n_eyes'] if col in df.columns]
    
    df_processed = df[feature_cols].copy()
    
    # Convert categorical to numeric with safe fallback
    mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
    for col in categorical_cols:
        mapped = df_processed[col].map(mapping)
        if mapped.isna().all():
            fill_value = 0
        else:
            mode_series = mapped.mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else 0
        df_processed[col] = mapped.fillna(fill_value)
    
    # Fill remaining NaN
    df_processed = df_processed.fillna(0)
    
    return df_processed.values

print("✅ Preprocessing functions defined!")

# %%
# Create sequences
print("🔄 Creating sequences...")

samples = []
labels = []

sample_indices = X_train['sample_index'].unique()

for sample_idx in sample_indices:
    # Get all time steps for this sample
    sample_data = X_train[X_train['sample_index'] == sample_idx].copy()
    sample_data = sample_data.sort_values('time')
    
    # Preprocess features
    features = preprocess_features(sample_data)
    samples.append(features)
    
    # Get label
    label = y_train[y_train['sample_index'] == sample_idx]['label'].iloc[0]
    labels.append(label)

print(f"✅ Created {len(samples)} sequences")
print(f"   Sequence length range: {min(s.shape[0] for s in samples)} - {max(s.shape[0] for s in samples)} time steps")
print(f"   Features per time step: {samples[0].shape[1]}")

# Scale features across all time steps to stabilize training
print("\n⚖️ Scaling features with StandardScaler...")
scaler = StandardScaler()
scaler.fit(np.vstack(samples))
samples = [scaler.transform(sample) for sample in samples]
print("✅ Scaling complete!")

# %%
# Pad sequences to uniform length
max_len = max([s.shape[0] for s in samples])
n_features = samples[0].shape[1]

print(f"\n📏 Padding sequences to length: {max_len}")

X_padded = np.zeros((len(samples), max_len, n_features))
for i, sample in enumerate(samples):
    X_padded[i, :sample.shape[0], :] = sample

print(f"✅ Padded data shape: {X_padded.shape}")
print(f"   Format: (samples, time_steps, features)")

# %%
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

class_counts = pd.Series(y_encoded).value_counts().sort_index()
print("\n🏷️ Label Encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"   {label} → {i}")
print("\nClass counts before balancing:")
for cls, count in class_counts.items():
    print(f"   Class {cls} ({label_encoder.classes_[cls]}): {count}")

# %%
# Create train/validation split (80/20)
print("\n✂️ Creating train/validation split...")

X_train_seq, X_val_seq, y_train_enc, y_val_enc = train_test_split(
    X_padded, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"✅ Training set: {X_train_seq.shape[0]} samples")
print(f"✅ Validation set: {X_val_seq.shape[0]} samples")

sequence_length = X_train_seq.shape[1]
n_features = X_train_seq.shape[2]

# Show class distribution before balancing
train_dist_before = pd.Series(y_train_enc).value_counts().sort_index()
val_dist = pd.Series(y_val_enc).value_counts().sort_index()

print("\n📊 Class distribution (original training split):")
for i, label in enumerate(label_encoder.classes_):
    print(f"   {label}: Train={train_dist_before.get(i, 0)}, Val={val_dist.get(i, 0)}")

# Keep original training data for resampling experiments
X_train_base = X_train_seq.astype(np.float32)
y_train_base = y_train_enc.copy()

print("\n🧪 Stored base training tensors for balancing strategies.")



# %%
# Custom F1-Score Metric for Keras
class F1Score(keras.metrics.Metric):
    """
    Custom F1-Score metric for Keras
    Computes macro-averaged F1-score for multi-class classification
    """
    def __init__(self, name='f1_score', num_classes=3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        
        y_true_one_hot = tf.cast(tf.one_hot(y_true, depth=self.num_classes), tf.float32)
        y_pred_one_hot = tf.cast(tf.one_hot(y_pred_classes, depth=self.num_classes), tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1, 1]), tf.float32)
            y_true_one_hot *= sample_weight
            y_pred_one_hot *= sample_weight
        
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f1)
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros((self.num_classes,)))
        self.false_positives.assign(tf.zeros((self.num_classes,)))
        self.false_negatives.assign(tf.zeros((self.num_classes,)))

print("✅ F1-Score metric class defined!")

# %%
def make_sparse_focal_loss(gamma=1.5, alpha=None):
    """Build a sparse categorical focal loss with optional per-class alpha weights."""
    alpha_tensor = None
    if alpha is not None:
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        if alpha_tensor is not None:
            class_weights = tf.gather(alpha_tensor, y_true)
            class_weights = tf.expand_dims(class_weights, axis=-1)
            focal_weight *= class_weights
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        per_sample_loss = tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
        return per_sample_loss

    return loss_fn

print("✅ Sparse focal loss factory defined!")

# %%
def build_model(
    sequence_length, 
    n_features, 
    n_classes, 
    lstm_units=(128, 64, 32), 
    dense_units=(64, 32), 
    dropout_lstm=0.3, 
    dropout_dense=0.3, 
    learning_rate=0.001, 
    gamma=2.0, 
    alpha=None
):
    """
    Build deep LSTM model for time series classification
    All weights randomly initialized - NO pre-training!
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(sequence_length, n_features), name='input_sequence'))
    model.add(layers.Masking(mask_value=0.0, name='mask_padding'))

    for idx, units in enumerate(lstm_units):
        return_sequences = idx < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_sequences, name=f'lstm_layer_{idx + 1}'))
        model.add(layers.Dropout(dropout_lstm))
        model.add(layers.BatchNormalization())

    for idx, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{idx + 1}'))
        model.add(layers.Dropout(dropout_dense))

    model.add(layers.Dense(n_classes, activation='softmax', name='output'))

    focal_loss_fn = make_sparse_focal_loss(gamma=gamma, alpha=alpha)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss_fn,
        metrics=['accuracy', F1Score(num_classes=n_classes)]
    )

    return model

# %%
# Model metadata
n_classes = len(label_encoder.classes_)
print("🏗️ Model metadata prepared.")
print(f"Number of classes: {n_classes}")
print(f"Sequence length: {sequence_length}")
print(f"Features per timestep: {n_features}")

# %%
# Define callbacks factory
def create_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=0.5,
            patience=5,
            mode='max',
            min_lr=1e-6,
            verbose=0
        )
    ]

print("✅ Callback factory ready!")

# %%
def apply_smote_resampling(X_train, y_train, smote_params):
    smote = SMOTE(**smote_params)
    X_flat = X_train.reshape(X_train.shape[0], -1)
    X_res_flat, y_res = smote.fit_resample(X_flat, y_train)
    X_res = X_res_flat.reshape(-1, sequence_length, n_features)
    return X_res.astype(np.float32), y_res

def compute_alpha_vector(y_labels, n_classes):
    present_classes = np.unique(y_labels)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=present_classes, y=y_labels)
    alpha = np.ones(n_classes, dtype=np.float32)
    for cls, weight in zip(present_classes, weights):
        alpha[int(cls)] = float(weight)
    return alpha

def run_experiment(smote_params, model_params, epochs=60, batch_size=32, return_model=False):
    tf.keras.backend.clear_session()
    X_resampled, y_resampled = apply_smote_resampling(X_train_base, y_train_base, smote_params)
    alpha_vec = compute_alpha_vector(y_resampled, n_classes)
    callbacks = create_callbacks()
    model = build_model(
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        alpha=alpha_vec,
        **model_params
    )
    history = model.fit(
        X_resampled, y_resampled,
        validation_data=(X_val_seq, y_val_enc),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    best_val_f1 = float(max(history.history['val_f1_score']))
    best_epoch = int(np.argmax(history.history['val_f1_score']))
    y_pred = np.argmax(model.predict(X_val_seq, verbose=0), axis=1)
    f1_macro = f1_score(y_val_enc, y_pred, average='macro')
    f1_weighted = f1_score(y_val_enc, y_pred, average='weighted')
    f1_per_class = f1_score(y_val_enc, y_pred, average=None)
    result = {
        'smote_params': smote_params,
        'model_params': model_params,
        'epochs_trained': len(history.history['loss']),
        'best_epoch': best_epoch,
        'history': history.history,
        'val_f1_macro': f1_macro,
        'val_f1_weighted': f1_weighted,
        'val_f1_keras_best': best_val_f1,
        'val_accuracy': float(history.history['val_accuracy'][best_epoch]),
        'f1_per_class': f1_per_class
    }
    if return_model:
        result['model'] = model
        result['X_resampled_shape'] = X_resampled.shape
    return result

# %%
# %% ⚡ OPTIMIZED GRID SEARCH (parallel + filtered)
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd

print("🚀 Starting optimized grid search...")

# -----------------------
# 1. Define search space
# -----------------------
smote_param_grid = [
    {'k_neighbors': 3, 'sampling_strategy': 'not majority', 'random_state': 42},
    {'k_neighbors': 5, 'sampling_strategy': 0.7, 'random_state': 42},
]

model_param_grid = [
    {'lstm_units': (64, 32), 'dropout_lstm': 0.2, 'dropout_dense': 0.2, 'learning_rate': 3e-4, 'gamma': 1.5},
    {'lstm_units': (128, 64, 32), 'dropout_lstm': 0.3, 'dropout_dense': 0.3, 'learning_rate': 7e-4, 'gamma': 2.0},
    {'lstm_units': (256, 128, 64), 'dropout_lstm': 0.35, 'dropout_dense': 0.3, 'learning_rate': 1e-3, 'gamma': 2.5},
    {'lstm_units': (160, 80, 32), 'dropout_lstm': 0.25, 'dropout_dense': 0.25, 'learning_rate': 5e-4, 'gamma': 1.8}
 ]

training_params = {'epochs': 60, 'batch_size': 32}

# --------------------------------------------
# 2. Filter out unrealistic or redundant combos
# --------------------------------------------
valid_combos = []
n_classes = len(np.unique(y_train_base))
for smote_params, model_params in product(smote_param_grid, model_param_grid):
    strategy = smote_params.get('sampling_strategy', 'auto')
    if isinstance(strategy, float) and n_classes > 2:
        continue
    lr = model_params['learning_rate']
    dr = model_params['dropout_lstm']
    if (lr < 1e-5) or (lr > 2e-3):
        continue
    if (dr > 0.5) or (dr < 0.1):
        continue
    valid_combos.append((smote_params, model_params))

print(f"✅ {len(valid_combos)} valid combinations will be tested.\n")

# -----------------------------------
# 3. Parallel execution with joblib
# -----------------------------------
def run_and_score(smote_params, model_params):
    """Wrapper that executes one experiment and returns metrics."""
    result = run_experiment(smote_params, model_params, **training_params)
    # Combined score: prioritize F1 but include accuracy
    result["combined_score"] = (
        0.7 * result["val_f1_macro"] + 0.3 * result["val_accuracy"]
    )
    return result

grid_results = Parallel(n_jobs=-1, backend="loky")(
    delayed(run_and_score)(sm, md)
    for sm, md in tqdm(valid_combos, desc="Running experiments")
)

# -----------------------------------
# 4. Collect results
# -----------------------------------
grid_results_df = pd.DataFrame([
    {
        'smote_k_neighbors': r['smote_params']['k_neighbors'],
        'smote_strategy': r['smote_params'].get('sampling_strategy', 'auto'),
        'lstm_units': str(r['model_params']['lstm_units']),
        'dropout_lstm': r['model_params']['dropout_lstm'],
        'dropout_dense': r['model_params']['dropout_dense'],
        'learning_rate': r['model_params']['learning_rate'],
        'gamma': r['model_params']['gamma'],
        'val_f1_macro': r['val_f1_macro'],
        'val_f1_weighted': r['val_f1_weighted'],
        'val_accuracy': r['val_accuracy'],
        'combined_score': r['combined_score'],
        'best_epoch': r['best_epoch']
    }
    for r in grid_results
])

grid_results_df = grid_results_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
display(grid_results_df)

# -----------------------------------
# 5. Select and report best result
# -----------------------------------
best_result = grid_results_df.iloc[0]
print("\n🥇 Best configuration found:")
print(best_result)

# Store best configuration for retraining
best_smote_params = next(r['smote_params'] for r in grid_results if r['combined_score'] == best_result['combined_score'])
best_model_params = next(r['model_params'] for r in grid_results if r['combined_score'] == best_result['combined_score'])

print(f"\nMacro F1: {best_result['val_f1_macro']:.4f} | "
      f"Accuracy: {best_result['val_accuracy']:.4f} | "
      f"Gamma: {best_result['gamma']} | "
      f"LR: {best_result['learning_rate']}")
print("\n✅ Optimized grid search completed successfully.")

# %%
# Retrain best configuration to capture full artifacts
print("🎯 Retraining best configuration for detailed evaluation...")
refit_epochs = 80
best_run = run_experiment(
    best_smote_params,
    best_model_params,
    epochs=refit_epochs,
    batch_size=training_params['batch_size'],
    return_model=True
)
best_model = best_run.pop('model')
best_history = best_run['history']
best_f1_per_class = best_run['f1_per_class']
print(
    f"Macro F1: {best_run['val_f1_macro']:.4f} | "
    f"Weighted F1: {best_run['val_f1_weighted']:.4f} | "
    f"Accuracy: {best_run['val_accuracy']:.4f}"
)

# %%
# Visualize training history of the best configuration
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Loss
axes[0].plot(best_history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(best_history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(best_history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(best_history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# F1 score
axes[2].plot(best_history['f1_score'], label='Training F1', linewidth=2)
axes[2].plot(best_history['val_f1_score'], label='Validation F1', linewidth=2)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('F1 Score', fontsize=12)
axes[2].set_title('F1 Score Over Time', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final metrics
print("\n📊 Final Training Metrics (Best Configuration):")
print("=" * 50)
print(f"Training Loss: {best_history['loss'][-1]:.4f}")
print(f"Training Accuracy: {best_history['accuracy'][-1]:.4f}")
print(f"Validation Loss: {best_history['val_loss'][-1]:.4f}")
print(f"Validation Accuracy: {best_history['val_accuracy'][-1]:.4f}")
print(f"Validation F1: {best_history['val_f1_score'][-1]:.4f}")

# %%
# Make predictions on validation set with the best model
print("🔮 Evaluating best model on validation set...\n")

y_pred_probs = best_model.predict(X_val_seq)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Calculate F1-scores
f1_macro = f1_score(y_val_enc, y_pred_classes, average='macro')
f1_weighted = f1_score(y_val_enc, y_pred_classes, average='weighted')
f1_per_class = f1_score(y_val_enc, y_pred_classes, average=None)

print("🎯 F1-SCORE RESULTS (PRIMARY METRIC):")
print("=" * 50)
print(f"Macro F1-Score:    {f1_macro:.4f} ⭐")
print(f"Weighted F1-Score: {f1_weighted:.4f}")
print("\nF1-Score per class:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label:15s}: {f1_per_class[i]:.4f}")
print("\n" + "=" * 50)

# Classification report
print("📋 Classification Report:")
print("=" * 70)
print(classification_report(y_val_enc, y_pred_classes, 
                          target_names=label_encoder.classes_))

# Overall accuracy
accuracy = np.mean(y_pred_classes == y_val_enc)
print(f"\n✨ Overall Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# %%
# Confusion matrix
cm = confusion_matrix(y_val_enc, y_pred_classes)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix - Validation Set', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Detailed confusion matrix analysis
print("\n🔍 Confusion Matrix Analysis:")
print("=" * 50)
for i, label in enumerate(label_encoder.classes_):
    total = cm[i].sum()
    correct = cm[i, i]
    print(f"{label}:")
    print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Misclassified: {total-correct}")

# %%
# Load test data
print("📂 Loading test data...")
X_test = pd.read_csv('pirate_pain_test.csv')

print(f"Test data shape: {X_test.shape}")
print(f"Number of test samples: {X_test['sample_index'].nunique()}")
print("\n📊 First few rows:")
display(X_test.head())

# %%
# Prepare test sequences
print("\n🔄 Preparing test sequences...")

test_samples = []
test_sample_indices = []

unique_test_samples = X_test['sample_index'].unique()

for sample_idx in unique_test_samples:
    # Get all time steps for this sample
    sample_data = X_test[X_test['sample_index'] == sample_idx].copy()
    sample_data = sample_data.sort_values('time')
    
    # Preprocess and scale features using the training scaler
    features = preprocess_features(sample_data)
    features_scaled = scaler.transform(features)
    test_samples.append(features_scaled.astype(np.float32))
    test_sample_indices.append(sample_idx)

print(f"✅ Created {len(test_samples)} test sequences")

# Pad or truncate to match the training sequence length
X_test_seq = np.zeros((len(test_samples), sequence_length, n_features), dtype=np.float32)
for i, sample in enumerate(test_samples):
    seq_len = sample.shape[0]
    if seq_len >= sequence_length:
        X_test_seq[i] = sample[:sequence_length]
    else:
        X_test_seq[i, :seq_len, :] = sample
        # Remaining timesteps stay zero-padded

print(f"✅ Test tensor shape: {X_test_seq.shape}")

# %%
# Make predictions
print("\n🔮 Generating predictions...")

y_test_probs = best_model.predict(X_test_seq)
y_test_classes = np.argmax(y_test_probs, axis=1)
y_test_labels = label_encoder.inverse_transform(y_test_classes)

print(f"✅ Generated {len(y_test_labels)} predictions")

# Show prediction distribution
pred_dist = pd.Series(y_test_labels).value_counts()
print("\n📊 Prediction distribution:")
print(pred_dist)

# Visualize
plt.figure(figsize=(10, 6))
colors_map = {'no_pain': 'green', 'low_pain': 'orange', 'high_pain': 'red'}
bar_colors = [colors_map.get(label, 'gray') for label in pred_dist.index]
plt.bar(pred_dist.index, pred_dist.values, color=bar_colors)
plt.xlabel('Pain Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Test Set Predictions Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Create submission
print("📝 Creating submission file...\n")

# Format sample indices as zero-padded 3-digit strings
formatted_indices = [f"{int(idx):03d}" for idx in test_sample_indices]

submission = pd.DataFrame({
    'sample_index': formatted_indices,
    'label': y_test_labels
})

# Save as comma-separated file
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)

print("✅ Submission file created: submission.csv")
print(f"   Format: Comma-separated (CSV)")
print(f"   Rows: {len(submission)}")
print(f"\n📋 First 15 predictions:")
print("=" * 40)
display(submission.head(15))

print("\n✨ Submission ready for upload!")


