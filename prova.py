# %% [markdown]
# # 🏴‍☠️ Pain Pirate Analysis - Pipeline Completa TensorFlow
# 
# Pipeline end-to-end con **tutte le 7 ADVICE del professore integrate nel codice**.
# 
# ## Dataset
# - **pirate_pain_train.csv**: 105,760 righe = 661 samples × 160 timesteps
# - **pirate_pain_train_labels.csv**: 661 labels (no_pain, low_pain, high_pain)
# - **Features**: 38 (4 pain_survey + 3 categorical + 31 joints)
# - **Classe dominante**: no_pain (511) - dataset **sbilanciato**!
# 
# ## ADVICE Integrate
# 1. ✅ **11/11 - Autocorrelazione**: Window size basata sui dati
# 2. ✅ **12/11 - Time Features**: Encoding ciclico temporale
# 3. ✅ **13/11 - Conv1D+LSTM**: Architettura ibrida
# 4. ✅ **10/11 - Gradient Clipping**: Stabilizza training
# 5. ✅ **09/11 - Label Smoothing**: Loss con smoothing
# 6. ✅ **08/11 - Class Weighting**: Gestisce sbilanciamento
# 7. ✅ **07/11 - Embeddings**: Features categoriche
# 

# %%
# Core libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Stats and ML
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Set seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')
print('✅ Environment ready!')

# %% [markdown]
# ## 1. Caricamento Dati

# %%
# Load dataset
X_train = pd.read_csv('pirate_pain_train.csv')
y_train = pd.read_csv('pirate_pain_train_labels.csv')

print('📊 Dataset Shape:')
print(f'  Features: {X_train.shape}')
print(f'  Labels: {y_train.shape}')
print(f'  Samples: {X_train["sample_index"].nunique()}')
print(f'  Timesteps/sample: {X_train.groupby("sample_index").size().iloc[0]}')

# Feature groups
pain_survey_cols = [c for c in X_train.columns if 'pain_survey' in c]
categorical_cols = ['n_legs', 'n_hands', 'n_eyes']
joint_cols = [c for c in X_train.columns if 'joint_' in c]

print(f'\n📋 Features: {len(pain_survey_cols)} pain_survey + {len(categorical_cols)} categorical + {len(joint_cols)} joints')

# ADVICE 08/11: Check class imbalance
print(f'\n🏷️ Labels (IMBALANCED - need class weighting):')
for label, count in y_train['label'].value_counts().items():
    print(f'  {label}: {count} ({100*count/len(y_train):.1f}%)')

# %% [markdown]
# ## 2. ADVICE 11/11: Determinare WINDOW_SIZE
# 
# *"Its own echo, the series sings."*
# 
# Usiamo autocorrelazione per scegliere window size basata sui dati.

# %%
# ADVICE 11/11: Analyze autocorrelation to determine optimal window
print('🔍 Analyzing autocorrelation...')
samples_analyze = X_train['sample_index'].unique()[:10]
key_features = joint_cols[:6]

optimal_lags = {}
for feature in key_features:
    sample_lags = []
    for sid in samples_analyze:
        data = X_train[X_train['sample_index']==sid][feature].values
        if len(data) >= 50:
            max_lags = min(len(data)//2-1, 80)
            acf_vals = acf(data, nlags=max_lags)
            sig_bound = 1.96/np.sqrt(len(data))
            for lag in range(1, len(acf_vals)):
                if abs(acf_vals[lag]) < sig_bound:
                    sample_lags.append(lag)
                    break
            else:
                sample_lags.append(max_lags)
    if sample_lags:
        optimal_lags[feature] = int(np.median(sample_lags))

if optimal_lags:
    suggested = int(np.median(list(optimal_lags.values())))
    WINDOW_SIZE = max(min(suggested, 100), 40)
else:
    WINDOW_SIZE = 60

WINDOW_STRIDE = WINDOW_SIZE // 2

print(f'✅ WINDOW_SIZE from autocorrelation: {WINDOW_SIZE}')
print(f'   STRIDE: {WINDOW_STRIDE}')
print(f'💡 ADVICE 11/11: Data-driven window size!')

# %% [markdown]
# ## 3. Preprocessing con ADVICE 07/11 e 12/11
# 
# **ADVICE 07/11**: Map categorical per embeddings  
# **ADVICE 12/11**: Aggiungi time features ciclici

# %%
# ADVICE 07/11: Map categorical features
cat_map = {
    'n_legs': {'two': 0, 'one+peg_leg': 1},
    'n_hands': {'two': 0, 'one+hook_hand': 1},
    'n_eyes': {'two': 0, 'one+eye_patch': 1}
}

X_proc = X_train.copy()
for col, mapping in cat_map.items():
    X_proc[col] = X_proc[col].map(mapping).fillna(0).astype(int)

# ADVICE 12/11: Add cyclical time features
max_time = X_proc['time'].max()
X_proc['time_sin'] = np.sin(2*np.pi*X_proc['time']/max_time)
X_proc['time_cos'] = np.cos(2*np.pi*X_proc['time']/max_time)
X_proc['time_norm'] = X_proc['time']/max_time

print('✅ Preprocessing done:')
print('   - ADVICE 07/11: Categorical mapped')
print('   - ADVICE 12/11: Time features (sin, cos, norm) added')
print(f'   Shape: {X_proc.shape}')

# %% [markdown]
# ## 4. Creazione Finestre

# %%
# Create sliding windows
def create_windows(df, sample_idx, window_size, stride):
    sample = df[df['sample_index']==sample_idx].sort_values('time')
    feat_cols = [c for c in sample.columns if c not in ['sample_index','time']]
    features = sample[feat_cols].values
    
    windows = []
    for start in range(0, max(1, len(features)-window_size+1), stride):
        end = min(start+window_size, len(features))
        win = features[start:end]
        if len(win) < window_size:
            pad = np.zeros((window_size-len(win), win.shape[1]))
            win = np.vstack([win, pad])
        windows.append(win)
    return windows

print('🔄 Creating windows...')
all_windows = []
all_labels = []

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train['label'])

for sid, label in zip(y_train['sample_index'], y_encoded):
    wins = create_windows(X_proc, sid, WINDOW_SIZE, WINDOW_STRIDE)
    all_windows.extend(wins)
    all_labels.extend([label]*len(wins))

X_windows = np.array(all_windows, dtype=np.float32)
y_windows = np.array(all_labels, dtype=np.int32)

print(f'✅ Windows: {X_windows.shape}')
print(f'   Labels: {y_windows.shape}')

# %%
# Split and normalize
X_tr, X_val, y_tr, y_val = train_test_split(
    X_windows, y_windows, test_size=0.2, random_state=SEED, stratify=y_windows
)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

print(f'📊 Split: Train {X_tr.shape}, Val {X_val.shape}')

# ADVICE 08/11: Compute class weights
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
class_weights_dict = {i: w for i, w in enumerate(class_weights_array)}

print(f'\n⚖️ ADVICE 08/11 - Class Weights:')
for i, w in class_weights_dict.items():
    print(f'   {label_encoder.classes_[i]}: {w:.3f}')

# %% [markdown]
# ## 5. ADVICE 13/11: Conv1D + LSTM
# 
# *"A pattern in time, like a pattern in space it is."*

# %%
# ADVICE 13/11: Build Conv1D + LSTM model
def build_conv_lstm_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Conv1D blocks
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='Conv_LSTM')

n_features = X_tr.shape[2]
n_classes = len(label_encoder.classes_)

model = build_conv_lstm_model((WINDOW_SIZE, n_features), n_classes)

print('✅ ADVICE 13/11: Conv1D + LSTM created')
print(f'   Input: ({WINDOW_SIZE}, {n_features})')
print(f'   Output: {n_classes} classes')
model.summary()

# %%
# Custom macro F1 metric so we can monitor validation F1 directly
class MacroF1(keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(shape=(num_classes,), initializer='zeros', name='tp')
        self.false_positives = self.add_weight(shape=(num_classes,), initializer='zeros', name='fp')
        self.false_negatives = self.add_weight(shape=(num_classes,), initializer='zeros', name='fn')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_true.shape.rank > 1:
            y_true_labels = tf.argmax(y_true, axis=-1)
        else:
            y_true_labels = tf.cast(y_true, tf.int32)
        y_pred_labels = tf.argmax(y_pred, axis=-1)

        y_true_one_hot = tf.one_hot(y_true_labels, depth=self.num_classes, dtype=tf.float32)
        y_pred_one_hot = tf.one_hot(y_pred_labels, depth=self.num_classes, dtype=tf.float32)

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
        f1 = 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f1)

    def reset_state(self):
        for var in (self.true_positives, self.false_positives, self.false_negatives):
            var.assign(tf.zeros_like(var))

# %% [markdown]
# ## 5.5. 🔍 Grid Search per Ottimizzazione Iperparametri
# 
# **Obiettivo**: Trovare automaticamente la combinazione di iperparametri che **massimizza F1 Macro Score**.
# 
# Strategia:
# - Testa sistematicamente diverse combinazioni di iperparametri
# - **F1 Macro** come metrica obiettivo (metrica della challenge)
# - Le ADVICE (gradient clipping, class weighting) sono **sempre applicate**
# - Early stopping per efficienza
# 
# **Nota**: Grid search può richiedere tempo. Per test rapidi, riduci il numero di combinazioni in `param_grid`.

# %%
from itertools import product
from joblib import Parallel, delayed
import time

param_grid = {
    # capacità della parte convoluzionale
    'conv_filters': [32, 64, 128],      # 96 invece di 128 per stare un filo più leggeri

    # capacità della parte ricorrente
    'lstm_units': [32, 64, 128, 192],      # 192 spesso basta, 256 inizia a essere pesante

    # regularization
    'dropout': [0.1, 0.2, 0.3],     # 0.2–0.3 di solito è la zona buona

    # ottimizzatore
    'learning_rate': [3e-4, 6e-4, 1e-3],   # meglio in scala log che 0.0005 / 0.001 secca

    # label smoothing (piccolo, visto che sono 3 classi)
    'label_smoothing': [0.0, 0.05, 0.1],

    # batch size (dipende dalla RAM, ma per serie corte va bene)
    'batch_size': [16, 32, 64],
}

# Generate all combinations
param_names = list(param_grid.keys())
param_values = list(param_grid.values())
param_combinations = list(product(*param_values))

print("🔍 Grid Search Configuration:")
print(f"   Parameters to test: {param_names}")
print(f"   Total combinations: {len(param_combinations)}")
print(f"   Estimated time depends on n_jobs (parallel execution)\n")

# Prepare one-hot targets once for label smoothing
y_tr_grid = keras.utils.to_categorical(y_tr, n_classes)
y_val_grid = keras.utils.to_categorical(y_val, n_classes)

def build_model_for_grid(input_shape, num_classes, conv_filters, lstm_units, dropout):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(conv_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(conv_filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def run_grid_experiment(param_tuple):
    params = dict(zip(param_names, param_tuple))
    conv_filters = params['conv_filters']
    lstm_units = params['lstm_units']
    dropout = params['dropout']
    learning_rate = params['learning_rate']
    label_smoothing = params['label_smoothing']
    batch_size = params['batch_size']

    tf.keras.backend.clear_session()
    model_grid = build_model_for_grid(
        (WINDOW_SIZE, n_features),
        n_classes,
        conv_filters,
        lstm_units,
        dropout
)

    model_grid.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            MacroF1(num_classes=n_classes)
        ]
)

    callbacks_local = [
        EarlyStopping(monitor='val_macro_f1', patience=5, mode='max', restore_best_weights=True, verbose=0)
]

    history = model_grid.fit(
        X_tr, y_tr_grid,
        validation_data=(X_val, y_val_grid),
        epochs=50,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=callbacks_local,
        verbose=0
)

    y_val_pred = model_grid.predict(X_val, verbose=0)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    f1_macro = f1_score(y_val, y_val_pred_classes, average='macro')

    result = {
        **params,
        'f1_macro': f1_macro,
        'epochs_trained': len(history.history['loss'])
    }

    tf.keras.backend.clear_session()
    return result

max_jobs = min(4, max(1, (os.cpu_count() or 2) // 2))
print(f"⚙️ Using {max_jobs} parallel workers (adjust max_jobs variable if needed)")

start_time = time.time()

grid_results = Parallel(n_jobs=max_jobs, backend='loky')(
    delayed(run_grid_experiment)(params)
    for params in tqdm(param_combinations, total=len(param_combinations), desc='Grid search')
)

elapsed_time = time.time() - start_time
print(f"✅ Grid search completed in {elapsed_time/60:.1f} minutes")

results_df = pd.DataFrame(grid_results).sort_values('f1_macro', ascending=False)

print('\n🏆 BEST CONFIGURATION:')
best_config = results_df.iloc[0]
print(f"   F1 Macro Score: {best_config['f1_macro']:.4f}\n")
print('   Parameters:')
for param in param_names:
    print(f"     {param}: {best_config[param]}")

print('\n📊 Grid Search Statistics:')
print(f"   Best F1 Macro: {results_df['f1_macro'].max():.4f}")
print(f"   Worst F1 Macro: {results_df['f1_macro'].min():.4f}")
print(f"   Mean F1 Macro: {results_df['f1_macro'].mean():.4f}")
print(f"   Std F1 Macro: {results_df['f1_macro'].std():.4f}")
print(f"   Improvement range: {results_df['f1_macro'].max() - results_df['f1_macro'].min():.4f}")

best_params = best_config[param_names].to_dict()

# %%
# Visualize grid search results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('🔍 Grid Search: Impact of Each Hyperparameter on F1 Macro Score', fontsize=16, fontweight='bold')

for idx, param in enumerate(param_names):
    ax = axes[idx // 3, idx % 3]

    grouped = results_df.groupby(param)['f1_macro'].agg(['mean', 'std', 'count'])

    x_pos = range(len(grouped))
    ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index)
    ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold')
    ax.set_ylabel('F1 Macro Score')
    ax.set_title(f'Impact of {param.replace("_", " ").title()}')
    ax.grid(axis='y', alpha=0.3)

    best_idx = grouped['mean'].idxmax()
    best_pos = list(grouped.index).index(best_idx)
    ax.get_children()[best_pos].set_color('green')
    ax.get_children()[best_pos].set_alpha(0.9)

plt.tight_layout()
plt.show()

# Top 10 configurations
print('\n📋 Top 10 Configurations (sorted by F1 Macro):')
print(results_df[param_names + ['f1_macro', 'epochs_trained']].head(10).to_string(index=False))

print('\n💡 Insights:')
print('   - Green bars show the best value for each hyperparameter')
print('   - Error bars show std deviation across different combinations')
print('   - Rankings use validation F1 macro only (accuracy not used)')

# %% [markdown]
# ### Usare la Best Configuration
# 
# Ora puoi:
# 1. **Continuare con i parametri di default** delle sezioni seguenti, oppure
# 2. **Modificare manualmente** le sezioni seguenti per usare `best_params`, oppure
# 3. **Ritrainare** un nuovo modello qui sotto con la best configuration per più epoche

# %% [markdown]
# ## 6. Compile con ADVICE 09/11 e 10/11
# 
# **ADVICE 09/11**: Label smoothing  
# **ADVICE 10/11**: Gradient clipping

# %%
tf.keras.backend.clear_session()

# %%
def build_model_for_grid(input_shape, num_classes, conv_filters, lstm_units, dropout):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(conv_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(conv_filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

model=build_model_for_grid((WINDOW_SIZE, n_features),n_classes,conv_filters=128, lstm_units=32, dropout=0.3)

# %%
# ADVICE 10/11: Optimizer with gradient clipping
# ADVICE 09/11: Loss with label smoothing
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # ADVICE 10/11
loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)  # ADVICE 09/11

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        MacroF1(num_classes=n_classes)
    ]
)

print('✅ Model compiled with:')
print('   - ADVICE 10/11: Gradient clipping (clipnorm=1.0)')
print('   - ADVICE 09/11: Label smoothing (0.1)')
print('   - Macro F1 metric to monitor validation performance')

# %%
# Convert labels to categorical for label smoothing
y_tr_cat = keras.utils.to_categorical(y_tr, n_classes)
y_val_cat = keras.utils.to_categorical(y_val, n_classes)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_macro_f1', patience=7, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_macro_f1', factor=0.5, patience=7, verbose=1, mode='max'),
    ModelCheckpoint('best_model_tf.h5', monitor='val_macro_f1', mode='max', save_best_only=True)
]

print('✅ Training ready (monitoring val_macro_f1)')

# %% [markdown]
# ## 7. Training con Tutte le ADVICE Integrate

# %%
print('🚀 Training with ALL ADVICE integrated:\n')
print('✅ 11/11 - Autocorrelation window')
print('✅ 12/11 - Time features (cyclical)')
print('✅ 13/11 - Conv1D + LSTM')
print('✅ 10/11 - Gradient clipping')
print('✅ 09/11 - Label smoothing')
print('✅ 08/11 - Class weighting')
print('✅ 07/11 - Categorical mapped')
print('✅ Monitoring val_macro_f1 for best model selection\n')

history = model.fit(
    X_tr, y_tr_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32,
    class_weight=class_weights_dict,  # ADVICE 08/11
    callbacks=callbacks,
    verbose=1
)

print('\n🎉 Training complete!')

# %%
# Plot history
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 7))

ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['accuracy'], label='Train')
ax2.plot(history.history['val_accuracy'], label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(history.history['macro_f1'], label='Train')
ax3.plot(history.history['val_macro_f1'], label='Val')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('F1 Score')
ax3.set_title('F1 Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Evaluation

# %%
# Evaluate
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)

print('📊 Classification Report:\n')
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_, digits=4))

# F1 Score
f1_macro = f1_score(y_val, y_pred, average='macro')
print(f'\n🎯 F1 Score (macro): {f1_macro:.4f}')

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🎓 Summary
# 
# Pipeline completa e funzionante con tutte le ADVICE integrate:
# 
# 1. **ADVICE 11/11**: Window size da autocorrelazione
# 2. **ADVICE 12/11**: Time features ciclici
# 3. **ADVICE 13/11**: Conv1D + LSTM
# 4. **ADVICE 10/11**: Gradient clipping (clipnorm=1.0)
# 5. **ADVICE 09/11**: Label smoothing (0.1)
# 6. **ADVICE 08/11**: Class weighting (balanced)
# 7. **ADVICE 07/11**: Categorical mapping
# 
# **Dataset**: 661 samples × 160 timesteps × 38 features  
# **Classes**: no_pain, low_pain, high_pain (sbilanciato)
# 
# ✅ **Pronto per essere eseguito end-to-end!** 🏴‍☠️

# %% [markdown]
# ## 9. Test Set Prediction e Submission

# %%
# Load test data
X_test = pd.read_csv('pirate_pain_test.csv')

print(f'📊 Test Data:')
print(f'   Shape: {X_test.shape}')
print(f'   Samples: {X_test["sample_index"].nunique()}')
print(f'   Timesteps/sample: {X_test.groupby("sample_index").size().iloc[0]}')

# %%
# Apply same preprocessing to test
X_test_proc = X_test.copy()

# ADVICE 07/11: Map categorical
for col, mapping in cat_map.items():
    X_test_proc[col] = X_test_proc[col].map(mapping).fillna(0).astype(int)

# ADVICE 12/11: Add time features
max_time_test = X_test_proc['time'].max()
X_test_proc['time_sin'] = np.sin(2*np.pi*X_test_proc['time']/max_time_test)
X_test_proc['time_cos'] = np.cos(2*np.pi*X_test_proc['time']/max_time_test)
X_test_proc['time_norm'] = X_test_proc['time']/max_time_test

print('✅ Test preprocessing done')

# %%
# Create windows for test and predict
print('🔄 Creating test windows and predicting...')

test_sample_indices = X_test['sample_index'].unique()
sample_predictions = {}  # Store predictions per sample

for sid in tqdm(test_sample_indices, desc='Predicting'):
    # Create windows for this sample
    windows = create_windows(X_test_proc, sid, WINDOW_SIZE, WINDOW_STRIDE)
    
    if len(windows) > 0:
        # Convert to array and normalize
        X_sample = np.array(windows, dtype=np.float32)
        X_sample = scaler.transform(
            X_sample.reshape(-1, X_sample.shape[-1])
        ).reshape(X_sample.shape)
        
        # Predict probabilities for all windows
        probs = model.predict(X_sample, verbose=0)
        
        # Aggregate: average probabilities across windows, then argmax
        avg_probs = probs.mean(axis=0)
        pred_class = np.argmax(avg_probs)
        
        sample_predictions[sid] = pred_class

print(f'✅ Predicted {len(sample_predictions)} samples')

# %%
# Create submission DataFrame
submission_data = []
for sid in sorted(sample_predictions.keys()):
    pred_class = sample_predictions[sid]
    pred_label = label_encoder.classes_[pred_class]
    submission_data.append({
        'sample_index': sid,
        'label': pred_label
    })

submission = pd.DataFrame(submission_data)

# Save submission
submission.to_csv('submission.csv', index=False)

print('✅ Submission created!')
print(f'   Shape: {submission.shape}')
print(f'   Columns: {list(submission.columns)}')
print(f'\n📊 Predicted label distribution:')
print(submission['label'].value_counts())
print(f'\n💾 Saved to: submission.csv')
display(submission.head(10))

# %% [markdown]
# ## 10. F1 Score Analysis e Ottimizzazione
# 
# Analizziamo come massimizzare l'F1 macro score sulla validation per migliorare le performance sul test.

# %%
# Analyze F1 score per class
from sklearn.metrics import f1_score, precision_score, recall_score

print('🎯 F1 Score Analysis on Validation Set\n')

# Get predictions
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate per-class metrics
f1_per_class = f1_score(y_val, y_pred, average=None)
precision_per_class = precision_score(y_val, y_pred, average=None)
recall_per_class = recall_score(y_val, y_pred, average=None)

print('📊 Per-Class Metrics:\n')
for i, label_name in enumerate(label_encoder.classes_):
    print(f'{label_name}:')
    print(f'  Precision: {precision_per_class[i]:.4f}')
    print(f'  Recall:    {recall_per_class[i]:.4f}')
    print(f'  F1 Score:  {f1_per_class[i]:.4f}')
    print()

# Overall F1 scores
f1_macro = f1_score(y_val, y_pred, average='macro')
f1_weighted = f1_score(y_val, y_pred, average='weighted')

print(f'🎯 Overall F1 Scores:')
print(f'   F1 Macro (challenge metric):    {f1_macro:.4f}')
print(f'   F1 Weighted:                     {f1_weighted:.4f}')

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Per-class F1 scores
ax1 = axes[0]
x_pos = np.arange(len(label_encoder.classes_))
ax1.bar(x_pos, f1_per_class, color=['green', 'orange', 'red'])
ax1.set_xticks(x_pos)
ax1.set_xticklabels(label_encoder.classes_, rotation=45)
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 Score per Class (Validation)')
ax1.axhline(y=f1_macro, color='blue', linestyle='--', label=f'Macro Avg: {f1_macro:.4f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Precision-Recall per class
ax2 = axes[1]
x_pos = np.arange(len(label_encoder.classes_))
width = 0.35
ax2.bar(x_pos - width/2, precision_per_class, width, label='Precision')
ax2.bar(x_pos + width/2, recall_per_class, width, label='Recall')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(label_encoder.classes_, rotation=45)
ax2.set_ylabel('Score')
ax2.set_title('Precision vs Recall per Class')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 💡 Tips per Massimizzare F1 Macro Score
# 
# ### 1. Class Weighting (già applicato ✅)
# Il class weighting aiuta con lo sbilanciamento, ma potrebbe non essere sufficiente.
# 
# ### 2. Threshold Tuning
# Invece di usare argmax, prova threshold diversi per ogni classe:
# ```python
# # Esempio: bias verso classi minoritarie
# thresholds = [0.3, 0.4, 0.5]  # per no_pain, low_pain, high_pain
# ```
# 
# ### 3. Ensemble di Predizioni
# Invece di media semplice sulle finestre, prova:
# - Media pesata (finestre centrali contano di più)
# - Voting (maggioranza tra finestre)
# - Max pooling delle probabilità
# 
# ### 4. Data Augmentation
# Per le classi minoritarie:
# - Noise injection
# - Time warping
# - Window shift augmentation
# 
# ### 5. Focal Loss
# Invece di label smoothing, usa focal loss per dare più peso agli esempi difficili:
# ```python
# # Focal loss = CE * (1-p)^gamma
# # Mette più enfasi su campioni classificati male
# ```
# 
# ### 6. Class-specific Hyperparameters
# Allena modelli separati o usa attention mechanism per dare più capacità alle classi minoritarie.
# 
# ### 7. Cross-Validation
# Usa K-fold CV per avere stime più robuste dell'F1:
# ```python
# from sklearn.model_selection import StratifiedKFold
# # 5-fold CV per vedere stabilità F1
# ```
# 
# ### 🎯 Focus sulle Classi Minoritarie
# F1 macro = media delle F1 per classe, quindi:
# - **low_pain** e **high_pain** hanno peso uguale a **no_pain**
# - Migliora recall su classi minoritarie
# - Monitora confusion matrix per vedere dove sbaglia
# 
# ### ⚠️ Overfitting Warning
# Se F1 validation >> F1 test:
# - Aumenta regolarizzazione (dropout, weight decay)
# - Riduci model complexity
# - Usa early stopping più aggressivo
# 


