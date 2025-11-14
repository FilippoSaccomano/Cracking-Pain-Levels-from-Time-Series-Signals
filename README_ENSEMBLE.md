# 🏴‍☠️ Ensemble con Optuna - Pipeline Automatizzata

## File Principale

**`pain_pirate_optuna_ensemble.ipynb`** - Notebook completo con Optuna + Ensemble

## Cosa Fa Questo Notebook

Questo notebook integra **Optuna optimization** con **ensemble approach** per trovare automaticamente i migliori modelli:

### Pipeline Completa
1. **Optuna Optimization**: cerca automaticamente i migliori iperparametri (es. 300 trial)
2. **Selezione Automatica**: seleziona i top N modelli dai trial di Optuna
3. **Training Ensemble**: allena i modelli selezionati con le loro configurazioni ottimali
4. **EWA Weighting**: pesa i modelli basandosi su validation F1
5. **Predizione**: genera submission con soft voting pesato

### Vantaggi Rispetto all'Approccio Manuale
- ✅ **Automatico**: non servono configurazioni hardcoded
- ✅ **Ottimizzato**: Optuna esplora lo spazio degli iperparametri
- ✅ **Flessibile**: cambia facilmente N_MODELS_ENSEMBLE (es. 5, 9, 15 modelli)
- ✅ **Migliori risultati**: ensemble dai trial migliori invece che da configurazioni fisse

### Metodo EWA (Exponentially Weighted Average)
I modelli vengono pesati in base al loro F1 score sul validation set:
```
peso_i = exp(-eta * (1 - F1_i))
```
Con `eta=15.0`, i modelli migliori ricevono peso maggiore.

## Come Usare

### 1. Requisiti
```python
torch, numpy, pandas, sklearn, scipy, tqdm, matplotlib, seaborn
```

### 2. Dati Necessari
- `pirate_pain_train.csv`
- `pirate_pain_train_labels.csv`
- `pirate_pain_test.csv`

### 3. Esecuzione
1. Apri `pain_pirate_optuna_ensemble.ipynb`
2. Esegui tutte le celle in ordine (`Run All`)
3. Attendi il completamento:
   - Optuna optimization: 2-6 ore (dipende da n_trials)
   - Ensemble training: 45-90 min su GPU
4. Il notebook genererà `submission_optuna_ensemble.csv`

### 4. Configurazione
Puoi modificare questi parametri nel notebook:
- `n_trials = 300`: numero di trial Optuna (aumenta per migliori risultati)
- `N_MODELS_ENSEMBLE = 9`: numero di modelli nell'ensemble (5-15 consigliato)

### 5. Struttura del Notebook

```
1. Caricamento dati
2. Preprocessing (autocorrelation, features categoriche, time features)
3. Creazione finestre sliding
4. Split train/validation
5. Definizione architettura Conv1D + BiLSTM
6. ⭐ Optuna optimization (300 trial, esplora spazio iperparametri)
7. ⭐ Selezione automatica top N modelli dai trial
8. ⭐ Training ensemble con configurazioni dei migliori trial
9. ⭐ Calcolo pesi EWA basato su validation F1
10. ⭐ Predizione test con ensemble pesato
11. Creazione submission
12. Visualizzazione risultati
```

## Output

### File Generati
- **`submission_optuna_ensemble.csv`**: Predizioni ensemble per il test set
- **`ensemble_optuna_results.png`**: Grafici F1 scores e pesi EWA

### Metriche Visualizzate
- F1 score individuale per ogni modello
- F1 score dell'ensemble
- Pesi EWA per ogni modello
- Distribuzione delle predizioni
- Grafici di confronto

## Vantaggi dell'Ensemble con Optuna

✅ **Automatizzato**: Optuna trova automaticamente i migliori iperparametri
✅ **Ensemble ottimizzato**: usa i migliori modelli trovati da Optuna invece di configurazioni manuali
✅ **Riduzione varianza**: media di N modelli riduce overfitting  
✅ **Voting pesato**: i modelli migliori hanno più influenza (EWA)
✅ **Flessibile**: cambia facilmente N_MODELS_ENSEMBLE (5-15 consigliato)
✅ **Robustezza**: meno sensibile a singoli modelli fallimentari  

## Tempi di Esecuzione

### Optuna Optimization (300 trial)
| Hardware | Tempo per trial | Tempo totale Optuna |
|----------|----------------|---------------------|
| CPU      | 2-4 min        | 10-20 ore          |
| GPU/CUDA | 30-60 sec      | 2.5-5 ore          |
| MPS (M1/M2/M3/M4) | 30-60 sec | 2.5-5 ore       |

### Ensemble Training (9 modelli)
| Hardware | Tempo per modello | Tempo totale Ensemble |
|----------|------------------|----------------------|
| CPU      | 20-40 min        | 3-6 ore             |
| GPU/CUDA | 5-10 min         | 45-90 min           |
| MPS (M1/M2/M3/M4) | 5-10 min  | 45-90 min        |

**Nota**: Optuna è la fase più lunga. Puoi ridurre `n_trials` (es. 100) per risultati più veloci.

## Note Tecniche

- Il notebook rileva automaticamente il device disponibile (MPS/CUDA/CPU)
- Optuna esplora spazio iperparametri: conv_filters, lstm_units, dropout, lr, weight_decay, label_smoothing, scheduler, gradient clipping
- Ogni modello usa early stopping basato su validation F1
- EWA (eta=15.0) pesa i modelli basandosi su F1 validation
- Pruning automatico di Optuna per fermare trial non promettenti

## Troubleshooting

**Problema**: CUDA out of memory  
**Soluzione**: Riduci batch_size o usa CPU

**Problema**: Optuna troppo lento  
**Soluzione**: Riduci `n_trials` da 300 a 100 o 50

**Problema**: Training troppo lento  
**Soluzione**: Usa GPU/MPS invece di CPU

**Problema**: F1 dell'ensemble peggiore dei singoli modelli  
**Soluzione**: Aumenta `N_MODELS_ENSEMBLE` o fai più trial Optuna

## File Necessari

✅ Solo `pain_pirate_optuna_ensemble.ipynb` è necessario  
✅ Nessun file Python esterno da importare  
✅ Pipeline completamente self-contained
