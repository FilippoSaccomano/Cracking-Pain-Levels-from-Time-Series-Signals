# %% [markdown]
# # Core libraries
# import os
# import random
# from collections import defaultdict
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
# 
# # Deep learning
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# 
# # Preprocessing and evaluation
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
# from sklearn.utils import class_weight
# 
# # Set random seeds for reproducibility
# os.environ["PYTHONHASHSEED"] = "42"
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)
# tf.keras.utils.set_random_seed(42)
# try:
#     tf.config.experimental.enable_op_determinism()
# except (AttributeError, RuntimeError):
#     pass
# 
# # Style settings
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette("husl")
# 
# print(f"TensorFlow version: {tf.__version__}")
# print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
# print("\n⚓ All libraries loaded successfully!")

# %%
# Core libraries
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Preprocessing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils import class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("\n⚓ All libraries loaded successfully!")

# %% [markdown]
# ## 1. Caricamento dei dati
# 
# Importiamo i file di training e impostiamo una panoramica generale su dimensioni e frequenza temporale.

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

# %% [markdown]
# ### Esplorazione delle prime righe
# 
# Visualizziamo un estratto di feature e label per verificare la qualità del caricamento.

# %%
# Display first few rows
print("\n📊 First few rows of training data:")
display(X_train.head())

print("\n🏷️ First few labels:")
display(y_train.head())

# %% [markdown]
# ### Distribuzione delle etichette
# 
# Analizziamo il bilanciamento tra le classi prima del preprocessing.

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

# %% [markdown]
# ### Analisi delle feature
# 
# Cataloghiamo le colonne rilevanti per le fasi successive di preprocessing.

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

# %% [markdown]
# ### Visualizzazione di un campione
# 
# Osserviamo le serie temporali delle misure per verificare andamenti e range dei segnali.

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

# %% [markdown]
# ## 2. Preprocessing a finestre scorrevoli
# 
# Definiamo le utility per pulizia dei segnali, generazione delle finestre e augmentations stocastiche.

# %%
# Sliding-window preprocessing and augmentation helpers
WINDOW_SIZE = 60
WINDOW_STRIDE = 30

CATEGORICAL_MAP = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
CATEGORICAL_COLS = ['n_legs', 'n_hands', 'n_eyes']
PAIN_SURVEY_COLS = [f'pain_survey_{i}' for i in range(1, 5)]
JOINT_COLS = [f'joint_{i:02d}' for i in range(31)]

def map_categorical_features(df):
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mapped = df[col].map(CATEGORICAL_MAP)
            if mapped.isna().all():
                df[col] = 0.0
            else:
                fill_value = float(mapped.dropna().median()) if not mapped.dropna().empty else 0.0
                df[col] = mapped.fillna(fill_value)
    return df

def interpolate_signals(df):
    joint_cols_present = [col for col in JOINT_COLS if col in df.columns]
    if joint_cols_present:
        df.loc[:, joint_cols_present] = df[joint_cols_present].replace(0, np.nan)
        df.loc[:, joint_cols_present] = df[joint_cols_present].interpolate(method='linear', limit_direction='both', axis=0)
        df.loc[:, joint_cols_present] = df[joint_cols_present].fillna(method='bfill').fillna(method='ffill').fillna(0.0)
    numeric_cols = [col for col in df.columns if col not in ['sample_index', 'time']]
    df.loc[:, numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both', axis=0)
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(0.0)
    return df

def preprocess_sample(sample_df):
    df = sample_df.sort_values('time').reset_index(drop=True).copy()
    df = map_categorical_features(df)
    df = interpolate_signals(df)
    feature_cols = [col for col in df.columns if col not in ['sample_index', 'time']]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    return features

def create_windows(sequence, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    windows = []
    n_steps, n_features = sequence.shape
    if n_steps <= window_size:
        padded = np.zeros((window_size, n_features), dtype=np.float32)
        padded[:n_steps] = sequence
        windows.append(padded)
        return windows
    start = 0
    while start + window_size <= n_steps:
        windows.append(sequence[start:start + window_size].astype(np.float32))
        start += stride
    if start < n_steps:
        tail = sequence[-window_size:]
        if tail.shape[0] < window_size:
            padded = np.zeros((window_size, n_features), dtype=np.float32)
            padded[:tail.shape[0]] = tail
            windows.append(padded)
        else:
            windows.append(tail.astype(np.float32))
    return windows

def _resample_sequence(seq, target_length):
    if seq.shape[0] == target_length:
        return seq.astype(np.float32)
    original_idx = np.linspace(0, seq.shape[0] - 1, num=seq.shape[0])
    target_idx = np.linspace(0, seq.shape[0] - 1, num=target_length)
    resampled = np.empty((target_length, seq.shape[1]), dtype=np.float32)
    for feature_idx in range(seq.shape[1]):
        resampled[:, feature_idx] = np.interp(target_idx, original_idx, seq[:, feature_idx])
    return resampled

def augment_noise(window, sigma=0.02, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=window.shape).astype(np.float32)
    return (window + noise).astype(np.float32)

def augment_scaling(window, scale_range=(0.9, 1.1), rng=None):
    rng = rng or np.random.default_rng()
    scale = rng.uniform(*scale_range)
    return (window * scale).astype(np.float32)

def augment_time_warp(window, stretch_range=(0.8, 1.2), rng=None):
    rng = rng or np.random.default_rng()
    stretch = rng.uniform(*stretch_range)
    target_length = max(2, int(round(window.shape[0] * stretch)))
    stretched = _resample_sequence(window, target_length)
    return _resample_sequence(stretched, window.shape[0])

def augment_dropout(window, drop_range=(0.05, 0.10), rng=None):
    rng = rng or np.random.default_rng()
    drop_ratio = rng.uniform(*drop_range)
    n_drop = max(1, int(round(drop_ratio * window.shape[0])))
    drop_indices = rng.choice(window.shape[0], size=n_drop, replace=False)
    augmented = window.copy()
    augmented[drop_indices] = 0.0
    return augmented.astype(np.float32)

AUGMENTATIONS = ['noise', 'time_warp', 'scaling', 'dropout']

def apply_random_augmentations(window, rng):
    augmented = window.copy()
    n_ops = int(rng.integers(1, len(AUGMENTATIONS) + 1))
    chosen = rng.choice(AUGMENTATIONS, size=n_ops, replace=False)
    for op in chosen:
        if op == 'noise':
            augmented = augment_noise(augmented, sigma=0.02, rng=rng)
        elif op == 'time_warp':
            augmented = augment_time_warp(augmented, stretch_range=(0.8, 1.2), rng=rng)
        elif op == 'scaling':
            augmented = augment_scaling(augmented, scale_range=(0.9, 1.1), rng=rng)
        elif op == 'dropout':
            augmented = augment_dropout(augmented, drop_range=(0.05, 0.10), rng=rng)
    return augmented.astype(np.float32)

print("✅ Sliding-window preprocessing utilities ready!")

# %% [markdown]
# ## 3. Costruzione del dataset a finestre
# 
# Convertiamo i segnali grezzi in tensori uniformi, applichiamo lo scaling e generiamo le finestre augmentate.

# %%
# %% 🔄 Build sliding-window dataset with augmentations
print("🔄 Building sliding-window dataset...")

rng = np.random.default_rng(42)
windows = []
window_labels = []

sample_indices = X_train['sample_index'].unique()

# --- 1️⃣ Creazione finestre ---
for sample_idx in sample_indices:
    sample_data = X_train[X_train['sample_index'] == sample_idx].copy()
    processed = preprocess_sample(sample_data)
    sample_windows = create_windows(processed, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
    label = y_train[y_train['sample_index'] == sample_idx]['label'].iloc[0]
    windows.extend(sample_windows)
    window_labels.extend([label] * len(sample_windows))

total_windows = len(windows)
feature_dim = windows[0].shape[1] if windows else 0
print(f"✅ Generated {total_windows} windows")
print(f"   Window size: {WINDOW_SIZE} | Stride: {WINDOW_STRIDE}")
print(f"   Feature dimensionality: {feature_dim}")

# --- 2️⃣ Encoding delle label ---
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(window_labels)

windows_array = np.stack(windows).astype(np.float32)
labels_array = encoded_labels.astype(np.int32)

# --- 3️⃣ Train/Validation split ---
X_train_seq, X_val_seq, y_train_enc, y_val_enc = train_test_split(
    windows_array,
    labels_array,
    test_size=0.2,
    random_state=42,
    stratify=labels_array
)
y_train_enc = y_train_enc.astype(np.int32)
y_val_enc = y_val_enc.astype(np.int32)

print(f"🧱 Training windows before augmentation: {X_train_seq.shape[0]}")
print(f"🧱 Validation windows: {X_val_seq.shape[0]}")

# --- 4️⃣ Normalizzazione con MinMaxScaler ---
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
n_samples, seq_len, n_features = X_train_seq.shape

X_train_seq = scaler.fit_transform(
    X_train_seq.reshape(-1, n_features)
).reshape(n_samples, seq_len, n_features)

X_val_seq = scaler.transform(
    X_val_seq.reshape(-1, n_features)
).reshape(X_val_seq.shape[0], seq_len, n_features)

print("✅ Normalizzazione MinMaxScaler applicata")

# --- 5️⃣ Data augmentation ---
train_count_before_aug = X_train_seq.shape[0]
augmented_windows = list(X_train_seq)
augmented_labels = list(y_train_enc)

for window, label in zip(X_train_seq, y_train_enc):
    if rng.random() < 0.5:
        augmented = apply_random_augmentations(window, rng)
        augmented_windows.append(augmented)
        augmented_labels.append(label)

X_train_seq = np.stack(augmented_windows).astype(np.float32)
y_train_enc = np.asarray(augmented_labels, dtype=np.int32)

print(f"✨ Training windows after augmentation: {X_train_seq.shape[0]} (Δ {X_train_seq.shape[0] - train_count_before_aug})")

# --- 6️⃣ Shape finali ---
sequence_length = WINDOW_SIZE
n_features = feature_dim

print(f"📐 Final tensor shapes → X_train: {X_train_seq.shape}, X_val: {X_val_seq.shape}")

# %% [markdown]
# ### Statistiche sulle etichette
# 
# Valutiamo la distribuzione delle classi dopo la fase di augmentazione e suddivisione.

# %%
# Label encoding overview
class_counts = pd.Series(y_train_enc).value_counts().sort_index()
print("\n🏷️ Label Encoding:")
for idx, label in enumerate(label_encoder.classes_):
    print(f"   {label} → {idx}")

print("\nClass distribution (training after augmentation):")
for cls_idx, count in class_counts.items():
    print(f"   Class {cls_idx} ({label_encoder.classes_[cls_idx]}): {count}")

val_counts = pd.Series(y_val_enc).value_counts().sort_index()
print("\nClass distribution (validation):")
for cls_idx, count in val_counts.items():
    print(f"   Class {cls_idx} ({label_encoder.classes_[cls_idx]}): {count}")

# %% [markdown]
# ### Persistenza dei tensori base
# 
# Manteniamo copie numpy dei tensori di train e validation per facilitare gli esperimenti successivi.

# %%
# Persist base tensors for downstream experiments
print("\n💾 Storing base training tensors...")

X_train_base = X_train_seq.astype(np.float32)
y_train_base = y_train_enc.copy()
X_val_base = X_val_seq.astype(np.float32)
y_val_base = y_val_enc.copy()

print(f"Training base shape: {X_train_base.shape}")
print(f"Validation base shape: {X_val_base.shape}")

# %% [markdown]
# ## 4. Metriche e callback
# 
# Definiamo la metrica personalizzata F1 macro per il monitoraggio durante l'addestramento.

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

# %% [markdown]
# ### Metadati per l'addestramento
# 
# Raccogliamo informazioni chiave sul problema per configurare i modelli.

# %%
# Model metadata
n_classes = len(label_encoder.classes_)
print("🏗️ Model metadata prepared.")
print(f"Number of classes: {n_classes}")
print(f"Sequence length: {sequence_length}")
print(f"Features per timestep: {n_features}")

# %% [markdown]
# ### Callback di addestramento
# 
# Utilizziamo early stopping e riduzione del learning rate per favorire la generalizzazione.

# %%
# Define callbacks factory
def create_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',
            mode='max',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

print("✅ Callback factory ready (EarlyStopping + LR scheduler + checkpoint)!")

# %% [markdown]
# ### Utility per i pesi di classe
# 
# Calcoliamo pesi bilanciati da fornire a Keras per mitigare lo sbilanciamento delle etichette.

# %%
# Training utilities aligned with sliding-window approach
def compute_balanced_class_weights(labels):
    """Calcola pesi bilanciati da passare a Keras per gestire lo sbilanciamento delle classi."""
    classes = np.unique(labels)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}

# %% [markdown]
# ## 5. Ricerca di modelli sequenziali
# 
# Configuriamo la griglia di esperimenti (LSTM/GRU/CNN-LSTM) e monitoriamo l'avanzamento con tqdm.

# %%
# %% ⚡ ENSEMBLE GRID SEARCH (LSTM, GRU, CNN-LSTM)
from contextlib import contextmanager
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score

print("🚀 Starting Ensemble Grid Search...")

# =====================================================
# 1. MODEL BUILDERS
# =====================================================
def build_lstm(sequence_length, n_features, n_classes,
               units=(128, 64), dense_units=(64,),
               dropout=0.3, dense_dropout=None, recurrent_dropout=0.0,
               bidirectional=False, use_layernorm=True):
    dense_dropout = dropout if dense_dropout is None else dense_dropout
    model = keras.Sequential()
    model.add(layers.Input(shape=(sequence_length, n_features)))

    for idx, unit_count in enumerate(units):
        return_sequences = idx < len(units) - 1
        lstm_layer = layers.LSTM(
            unit_count,
            return_sequences=return_sequences,
            recurrent_dropout=recurrent_dropout
        )
        if bidirectional:
            lstm_layer = layers.Bidirectional(lstm_layer)
        model.add(lstm_layer)
        if use_layernorm and return_sequences:
            model.add(layers.LayerNormalization())
        model.add(layers.Dropout(dropout))

    for d in dense_units:
        model.add(layers.Dense(d, activation='relu'))
        if dense_dropout > 0:
            model.add(layers.Dropout(dense_dropout))

    model.add(layers.Dense(n_classes, activation='softmax'))
    return model


def build_gru(sequence_length, n_features, n_classes, units=(128, 64), dropout=0.3):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.GRU(units[0], return_sequences=True),
        layers.Dropout(dropout),
        layers.GRU(units[1]),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model


def build_cnn_lstm(sequence_length, n_features, n_classes,
                   filters=64, kernel_size=3, lstm_units=64,
                   dense_units=(64,), dropout=0.3, dense_dropout=None,
                   recurrent_dropout=0.0, bidirectional=False,
                   use_layernorm=True, use_noise=True):
    dense_dropout = dropout if dense_dropout is None else dense_dropout
    model = keras.Sequential()

    # Input + Gaussian noise
    if use_noise:
        model.add(layers.GaussianNoise(0.01, input_shape=(sequence_length, n_features)))
    else:
        model.add(layers.Input(shape=(sequence_length, n_features)))

    # CNN block
    model.add(layers.Conv1D(filters, kernel_size, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))
    if use_layernorm:
        model.add(layers.LayerNormalization())
    model.add(layers.Dropout(dropout))

    # LSTM block
    if bidirectional:
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units, recurrent_dropout=recurrent_dropout)
        ))
    else:
        model.add(layers.LSTM(lstm_units, recurrent_dropout=recurrent_dropout))
    model.add(layers.Dropout(dropout))

    # Dense head
    for d in dense_units:
        model.add(layers.Dense(d, activation='relu'))
        model.add(layers.Dropout(dense_dropout))

    # Output
    model.add(layers.Dense(n_classes, activation='softmax'))
    return model


def build_bilstm(sequence_length, n_features, n_classes,
                 units=(256, 128), dropout=0.3, recurrent_dropout=0.0,
                 merge_mode='concat', dense_units=None, dense_dropout=None):
    dense_units = tuple(dense_units or ())
    dense_dropout = dropout if dense_dropout is None else dense_dropout
    model = keras.Sequential()
    model.add(layers.Input(shape=(sequence_length, n_features)))

    for idx, unit_count in enumerate(units):
        return_sequences = idx < len(units) - 1
        model.add(layers.Bidirectional(
            layers.LSTM(unit_count, return_sequences=return_sequences,
                        recurrent_dropout=recurrent_dropout),
            merge_mode=merge_mode
        ))
        model.add(layers.Dropout(dropout))
        if return_sequences:
            model.add(layers.BatchNormalization())

    for dense_count in dense_units:
        model.add(layers.Dense(dense_count, activation='relu'))
        if dense_dropout > 0:
            model.add(layers.Dropout(dense_dropout))

    model.add(layers.Dense(n_classes, activation='softmax'))
    return model


# =====================================================
# 2. MODEL WRAPPER
# =====================================================
def build_generic_model(model_type, **kwargs):
    if model_type == "LSTM":
        return build_lstm(**kwargs)
    if model_type == "GRU":
        return build_gru(**kwargs)
    if model_type == "CNN_LSTM":
        return build_cnn_lstm(**kwargs)
    if model_type == "BI_LSTM":
        return build_bilstm(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")


# =====================================================
# 3. PARAMETER GRID
# =====================================================
model_param_grid = [
    {
        'model_type': 'CNN_LSTM',
        'filters': 96,
        'kernel_size': 5,
        'lstm_units': 96,
        'dense_units': (128,),
        'dropout': 0.3,
        'recurrent_dropout': 0.1,
        'learning_rate': 7e-4,
        'bidirectional': True
    },
    {
        'model_type': 'CNN_LSTM',
        'filters': 64,
        'kernel_size': 3,
        'lstm_units': 128,
        'dense_units': (64,),
        'dropout': 0.25,
        'recurrent_dropout': 0.15,
        'learning_rate': 6e-4,
        'bidirectional': True
    },
    {
        'model_type': 'CNN_LSTM',
        'filters': 128,
        'kernel_size': 3,
        'lstm_units': 64,
        'dense_units': (128,),
        'dropout': 0.35,
        'recurrent_dropout': 0.1,
        'learning_rate': 5e-4,
        'bidirectional': False
    },
    {
        'model_type': 'CNN_LSTM',
        'filters': 96,
        'kernel_size': 7,
        'lstm_units': 96,
        'dense_units': (128,),
        'dropout': 0.3,
        'learning_rate': 8e-4,
        'bidirectional': False
    },
    {
        'model_type': 'CNN_LSTM',
        'filters': 80,
        'kernel_size': 3,
        'lstm_units': 80,
        'dense_units': (128, 64),
        'dropout': 0.3,
        'recurrent_dropout': 0.1,
        'learning_rate': 9e-4,
        'bidirectional': True,
        'use_noise': True,
        'use_layernorm': True
    },
]

training_params = {'epochs': 60, 'batch_size': 1024}
balanced_class_weights = compute_balanced_class_weights(y_train_base)


# =====================================================
# Helper: progress bar for joblib
# =====================================================
@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        tqdm_object.close()


# =====================================================
# 4. EXPERIMENT FUNCTION (improved)
# =====================================================
def run_single_experiment(model_params):
    # --- copia parametri locali ---
    model_params_local = model_params.copy()
    model_type = model_params_local.pop("model_type")
    learning_rate = model_params_local.pop("learning_rate", 1e-3)
    dropout = model_params_local.pop("dropout", 0.3)

    build_args = dict(
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes
    )

    # --- costruzione del modello dinamica ---
    model = build_generic_model(
        model_type,
        **build_args,
        dropout=dropout,
        **model_params_local
    )

    print(f"\n🧩 Training {model_type} | Params:")
    for k, v in model_params_local.items():
        print(f"   {k}: {v}")

    # --- ottimizzatore con clip gradiente (evita esplosioni LSTM) ---
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )

    # --- loss e metriche ---
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            F1Score(num_classes=n_classes, average='macro', name='val_f1_score')
        ]
    )

    # --- addestramento ---
    history = model.fit(
        X_train_base, y_train_base,
        validation_data=(X_val_seq, y_val_enc),
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        callbacks=create_callbacks() + [
            keras.callbacks.TerminateOnNaN()
        ],
        class_weight=balanced_class_weights,
        verbose=1
    )

    # --- valutazione finale ---
    val_pred_probs = model.predict(X_val_seq, verbose=0)
    y_pred = np.argmax(val_pred_probs, axis=1)

    f1_macro = f1_score(y_val_enc, y_pred, average='macro')
    acc = np.mean(y_val_enc == y_pred)

    # --- tracking delle metriche ---
    history_dict = history.history
    val_f1_history = history_dict.get('val_f1_score')
    best_epoch = int(np.argmax(val_f1_history)) if val_f1_history is not None else None
    best_val_f1 = float(max(val_f1_history)) if val_f1_history is not None else None

    # --- log risultato ---
    result = {
        'model_type': model_type,
        'model_config': {**model_params_local, 'dropout': dropout, 'learning_rate': learning_rate},
        'val_f1_macro': f1_macro,
        'val_accuracy': acc,
        'val_pred_probs': val_pred_probs.astype(np.float32),
        'epochs_trained': len(history_dict.get('loss', [])),
        'best_epoch': best_epoch,
        'val_f1_keras_best': best_val_f1
    }

    # --- pulizia memoria ---
    tf.keras.backend.clear_session()
    del model
    return result


# =====================================================
# 5. PARALLEL GRID SEARCH
# =====================================================
total_jobs = len(model_param_grid)
with tqdm_joblib(tqdm(total=total_jobs, desc="Training models", unit="model")):
    grid_results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(m) for m in model_param_grid
    )


# =====================================================
# 6. ENSEMBLE (Weighted Voting) — Improved
# =====================================================
print("\n🧠 Creating ensemble predictions (weighted voting)...")

preds = []
weights = []
model_scores = []

for res in grid_results:
    pred_probs = res.get('val_pred_probs')
    f1_val = res.get('val_f1_macro', 0.0)

    if pred_probs is None or np.isnan(f1_val):
        continue

    preds.append(pred_probs)
    weights.append(max(f1_val, 1e-6))  # evita divisione per 0
    model_scores.append(f1_val)

# --- sicurezza ---
if len(preds) == 0:
    raise RuntimeError("❌ Nessuna predizione valida trovata per l'ensemble.")

# --- normalizza pesi ---
weights = np.array(weights)
weights = weights / np.sum(weights)

# --- ensemble pesato ---
ensemble_pred = np.average(preds, axis=0, weights=weights)
ensemble_classes = np.argmax(ensemble_pred, axis=1)

# --- metriche ensemble ---
ensemble_f1 = f1_score(y_val_enc, ensemble_classes, average='macro')
ensemble_acc = np.mean(ensemble_classes == y_val_enc)

# --- statistiche sui modelli singoli ---
mean_f1 = np.mean(model_scores)
std_f1 = np.std(model_scores)
top3_f1 = sorted(model_scores, reverse=True)[:3]

# --- risultati ---
print("\n📊 ENSEMBLE SUMMARY")
print(f"Weighted voting using {len(preds)} models")
print(f"→ Mean F1 (single models): {mean_f1:.4f} ± {std_f1:.4f}")
print(f"→ Top 3 F1 individual: {top3_f1}")
print(f"\n🥇 ENSEMBLE PERFORMANCE")
print(f"F1-macro (ensemble): {ensemble_f1:.4f}")
print(f"Accuracy (ensemble): {ensemble_acc:.4f}")

# --- salva risultati intermedi ---
ensemble_results = {
    "ensemble_f1": ensemble_f1,
    "ensemble_acc": ensemble_acc,
    "model_scores": model_scores,
    "weights": weights.tolist(),
    "preds_shape": [p.shape for p in preds]
}


# =====================================================
# 7. RESULTS SUMMARY — Improved reporting
# =====================================================

# --- Crea tabella con risultati principali ---
grid_results_df = pd.DataFrame([
    {
        'model_type': r.get('model_type', '?'),
        'val_f1_macro': r.get('val_f1_macro', 0),
        'val_accuracy': r.get('val_accuracy', 0),
        'epochs_trained': r.get('epochs_trained', 0),
        'best_epoch': r.get('best_epoch', None),
        **{f"param_{k}": v for k, v in r.get('model_config', {}).items()}
    }
    for r in grid_results
])

# --- Ordina per F1 macro ---
grid_results_df = grid_results_df.sort_values('val_f1_macro', ascending=False).reset_index(drop=True)

# --- Mostra top risultati ---
print("\n📊 INDIVIDUAL MODEL RESULTS (sorted by F1)")
display(grid_results_df.head(10))

# --- Statistiche globali ---
mean_f1 = grid_results_df['val_f1_macro'].mean()
std_f1 = grid_results_df['val_f1_macro'].std()
best_f1 = grid_results_df['val_f1_macro'].max()
best_acc = grid_results_df.loc[0, 'val_accuracy']

print(f"\n📈 Mean F1 across all models: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"🏆 Best F1: {best_f1:.4f} | Best Accuracy: {best_acc:.4f}")
print(f"🧠 Ensemble F1: {ensemble_f1:.4f} | Ensemble Acc: {ensemble_acc:.4f}")

# --- Mostra il miglior modello completo ---
best_single_result_raw = max(grid_results, key=lambda r: r.get('val_f1_macro', 0))
best_single_result = best_single_result_raw.copy()
best_single_result.pop('val_pred_probs', None)
best_model_config = best_single_result.get('model_config', {}).copy()

print("\n🥇 BEST SINGLE MODEL DETAILS")
print(f"Type: {best_single_result.get('model_type')}")
print(f"Macro F1: {best_single_result.get('val_f1_macro'):.4f}")
print(f"Accuracy: {best_single_result.get('val_accuracy'):.4f}")
print(f"Epochs Trained: {best_single_result.get('epochs_trained')}")
print(f"Best Epoch: {best_single_result.get('best_epoch')}")
print("Hyperparameters:")
for k, v in best_model_config.items():
    print(f"  • {k}: {v}")

# --- Salva risultati se serve ---
results_summary = {
    "ensemble_f1": ensemble_f1,
    "ensemble_acc": ensemble_acc,
    "mean_f1": mean_f1,
    "std_f1": std_f1,
    "best_model": best_model_config,
    "grid_df": grid_results_df
}

# Esempio: salva i risultati in CSV
grid_results_df.to_csv("grid_results_summary.csv", index=False)
print("\n💾 Results saved to 'grid_results_summary.csv'")

# %% [markdown]
# ## 6. Rifinitura del modello migliore
# 
# Rialleniamo la migliore configurazione per più epoche e ne misuriamo le metriche di validazione.

# %%
# Retrain best configuration to capture full artifacts
print("🎯 Retraining best configuration for detailed evaluation...")
if 'best_single_result' not in globals():
    raise RuntimeError("Esegui prima la cella di grid search per determinare la configurazione migliore.")

refit_epochs = 80
best_model_config = best_single_result['model_config'].copy()
best_model_type = best_model_config.pop('model_type') 
best_learning_rate = best_model_config.pop('learning_rate', 1e-3)
best_dropout = best_model_config.pop('dropout', 0.3)
best_model_summary = {
    'model_type': best_model_type,
    **best_model_config,
    'dropout': best_dropout,
    'learning_rate': best_learning_rate
}
print(f"Configurazione selezionata: {best_model_summary}")

# Costruisci e riallena il modello con più epoche
best_model = build_generic_model(
    best_model_type,
    sequence_length=sequence_length,
    n_features=n_features,
    n_classes=n_classes,
    dropout=best_dropout,
    **best_model_config
 )
best_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', F1Score(num_classes=n_classes)]
 )

best_history_obj = best_model.fit(
    X_train_base,
    y_train_base,
    validation_data=(X_val_seq, y_val_enc),
    epochs=refit_epochs,
    batch_size=training_params['batch_size'],
    callbacks=create_callbacks(),
    class_weight=balanced_class_weights,
    verbose=0
 )
best_history = best_history_obj.history

# Calcola metriche sulla validation
y_val_pred = np.argmax(best_model.predict(X_val_seq, verbose=0), axis=1)
val_f1_macro = f1_score(y_val_enc, y_val_pred, average='macro')
val_f1_weighted = f1_score(y_val_enc, y_val_pred, average='weighted')
val_accuracy = np.mean(y_val_pred == y_val_enc)

print(
    f"Macro F1: {val_f1_macro:.4f} | "
    f"Weighted F1: {val_f1_weighted:.4f} | "
    f"Accuracy: {val_accuracy:.4f}"
)

# %% [markdown]
# ### Andamento dell'addestramento
# 
# Tracciamo loss, accuracy e F1 per monitorare il comportamento del modello rifinito.

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

# %% [markdown]
# ### Valutazione su validation
# 
# Calcoliamo metriche di classificazione e report completo sul validation set.

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

# %% [markdown]
# ### Confusion matrix e analisi errori
# 
# Ispezioniamo il dettaglio delle predizioni per classe per individuare pattern di errore.

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

# %% [markdown]
# ## 7. Preparazione del test set
# 
# Carichiamo i dati di test e ne verifichiamo struttura e campioni principali.

# %%
# Load test data
print("📂 Loading test data...")
X_test = pd.read_csv('pirate_pain_test.csv')

print(f"Test data shape: {X_test.shape}")
print(f"Number of test samples: {X_test['sample_index'].nunique()}")
print("\n📊 First few rows:")
display(X_test.head())

# %% [markdown]
# ### Generazione delle finestre di test
# 
# Applichiamo lo stesso preprocessing su ciascun campione del test set.

# %%
# Prepare test windows
print("\n🔄 Preparing test windows...")

test_windows = []
test_window_sample_indices = []

unique_test_samples = X_test['sample_index'].unique()

for sample_idx in unique_test_samples:
    sample_data = X_test[X_test['sample_index'] == sample_idx].copy()
    processed = preprocess_sample(sample_data)
    sample_windows = create_windows(processed, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
    for window in sample_windows:
        test_windows.append(window.astype(np.float32))
        test_window_sample_indices.append(sample_idx)

if not test_windows:
    raise ValueError("Nessuna finestra generata dal set di test.")

test_windows_array = np.stack(test_windows).astype(np.float32)
test_windows_flat = test_windows_array.reshape(-1, n_features)
test_windows_scaled = scaler.transform(test_windows_flat).reshape(-1, WINDOW_SIZE, n_features).astype(np.float32)

print(f"✅ Created {len(test_windows_scaled)} test windows from {len(unique_test_samples)} samples")
print(f"   Window shape: {test_windows_scaled.shape[1:]}")
print(f"   Unique sample indices: {len(set(test_window_sample_indices))}")

# %% [markdown]
# ### Predizione aggregata sul test
# 
# Otteniamo le probabilità per finestra, le aggreghiamo a livello di campione e analizziamo la distribuzione finale.

# %%
# Make predictions
print("\n🔮 Generating predictions...")

window_probs = best_model.predict(test_windows_scaled)

sample_probabilities = defaultdict(list)
for prob, sample_idx in zip(window_probs, test_window_sample_indices):
    sample_probabilities[sample_idx].append(prob)

aggregated_indices = []
aggregated_probs = []
for sample_idx in unique_test_samples:
    probs = np.stack(sample_probabilities[sample_idx])
    aggregated_indices.append(sample_idx)
    aggregated_probs.append(probs.mean(axis=0))

aggregated_probs = np.stack(aggregated_probs)
y_test_classes = np.argmax(aggregated_probs, axis=1)
y_test_labels = label_encoder.inverse_transform(y_test_classes)

print(f"✅ Generated predictions for {len(y_test_labels)} samples")

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

# %% [markdown]
# ### Esportazione della submission
# 
# Formattiamo gli indici e salviamo il file `submission.csv` conforme alle specifiche della competizione.

# %%
# Save submission
formatted_indices = [f"{int(idx):03d}" for idx in aggregated_indices]
submission_df = pd.DataFrame({
    'sample_index': formatted_indices,
    'label': y_test_labels
}).sort_values('sample_index').reset_index(drop=True)
submission_path = os.path.join(os.getcwd(), 'submission.csv')
submission_df.to_csv(submission_path, index=False)
print(f"✅ Submission saved to {submission_path}")


