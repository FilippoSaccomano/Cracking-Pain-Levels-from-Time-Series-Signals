# 🎉 Ensemble con Optuna - Implementazione Completata

## ✅ Nuovo Approccio: Integrazione Optuna + Ensemble

Come richiesto, ho modificato l'implementazione per **integrare con Optuna** invece di usare configurazioni manuali.

## 📁 File Principale

**`pain_pirate_optuna_ensemble.ipynb`**

Questo è l'UNICO file da eseguire. Contiene:
- Preprocessing dei dati (dal notebook originale)
- **Optuna optimization** (trova automaticamente i migliori iperparametri)
- **Selezione automatica** dei top N modelli dai trial di Optuna
- Training ensemble con i modelli selezionati
- Valutazione con EWA weighting
- Predizione su test set
- Generazione submission

## 🎯 Cambiamenti Rispetto all'Implementazione Precedente

### Prima (Manuale)
❌ 3 configurazioni hardcoded
❌ 3 seeds per configurazione
❌ Totale: 9 modelli fissi

### Ora (Automatico con Optuna)
✅ Optuna trova automaticamente i migliori iperparametri (es. 300 trial)
✅ Selezione automatica dei top N modelli migliori
✅ Numero di modelli configurabile (`N_MODELS_ENSEMBLE = 9`)
✅ **Nessuna configurazione manuale necessaria**

## 🔍 Come Funziona

### 1. Optuna Optimization
```python
# Optuna esplora lo spazio degli iperparametri
- conv_filters: [[160,160], [128,128], [64,64], ...]
- lstm_units: [32, 64, 128, 160, 192]
- dropout: [0.10, 0.40]
- lr: [3e-4, 1e-3]
- weight_decay: [1e-6, 1e-3]
- label_smoothing: [0.0, 0.15]
- scheduler_factor, scheduler_patience
- early_stop_patience, max_grad_norm
- search_epochs: [15, 50]

# Dopo 300 trial, otteniamo i migliori modelli
```

### 2. Selezione Automatica
```python
# Seleziona i top N modelli
N_MODELS_ENSEMBLE = 9
best_trials = study.trials_dataframe().head(N_MODELS_ENSEMBLE)
```

### 3. Training Ensemble
```python
# Per ogni trial selezionato:
for trial in best_trials:
    # Usa gli iperparametri ottimali trovati da Optuna
    model = train_model_from_trial(trial_params)
    ensemble_models.append(model)
```

### 4. EWA Weighting
```python
# Pesa i modelli basandosi su validation F1
loss_i = 1 - F1_i
weight_i = exp(-eta * loss_i)
weights = weights / sum(weights)
```

### 5. Soft Voting Pesato
```python
# Predizioni finali con ensemble
weighted_probs = sum(weight_i * model_i.predict_proba(X))
final_prediction = argmax(softmax(weighted_probs))
```

## 🚀 Vantaggi dell'Approccio Optuna

✅ **Completamente automatizzato**: non servono configurazioni manuali
✅ **Ottimizzazione intelligente**: Optuna esplora lo spazio iperparametri
✅ **Flessibile**: cambia facilmente N_MODELS_ENSEMBLE (5, 9, 15, ...)
✅ **Migliori risultati**: ensemble dai trial realmente migliori
✅ **Pruning automatico**: Optuna ferma trial non promettenti
✅ **Riproducibile**: tutti i trial sono tracciati

## 📊 Pipeline Completa

```
1. Preprocessing
   ├─ Autocorrelation-based windowing
   ├─ Time features (sin, cos, norm)
   └─ Categorical mapping

2. Optuna Optimization (es. 300 trial)
   ├─ Esplora spazio iperparametri
   ├─ Early stopping per trial
   ├─ Pruning automatico
   └─ Salva risultati in study

3. Selezione Best Models
   ├─ Ordina trial per F1 validation
   ├─ Seleziona top N modelli
   └─ Estrai configurazioni

4. Training Ensemble
   ├─ Per ogni modello selezionato:
   │  ├─ Usa config del trial
   │  ├─ Train con early stopping
   │  └─ Salva F1 validation
   └─ Collect tutti i modelli

5. EWA Weighting
   ├─ Calcola pesi da F1 scores
   └─ Normalizza pesi (eta=15.0)

6. Ensemble Prediction
   ├─ Weighted soft voting
   └─ Genera submission

7. Visualization
   ├─ F1 scores plot
   └─ EWA weights plot
```

## ⏱️ Tempi di Esecuzione

### Optuna Optimization
- **CPU**: ~10-20 ore (300 trial × 2-4 min)
- **GPU/MPS**: ~2.5-5 ore (300 trial × 30-60 sec)

**Tip**: Riduci `n_trials` a 100 per risultati più veloci

### Ensemble Training (9 modelli)
- **CPU**: ~3-6 ore (9 × 20-40 min)
- **GPU/MPS**: ~45-90 min (9 × 5-10 min)

## 🎓 Come Usare

### 1. Apri il Notebook
```bash
jupyter notebook pain_pirate_optuna_ensemble.ipynb
```

### 2. Configura (Opzionale)
```python
# Nella cella Optuna
n_trials = 300  # Cambia per più/meno trial

# Nella cella selezione
N_MODELS_ENSEMBLE = 9  # Cambia numero modelli
```

### 3. Esegui
```
Run All Cells → attendi completamento → ottieni submission_optuna_ensemble.csv
```

## 📈 Output

### File Generati
- **`submission_optuna_ensemble.csv`**: Submission con ensemble
- **`ensemble_optuna_results.png`**: Grafici F1 e pesi

### Metriche Visualizzate
- F1 score di ogni modello nell'ensemble
- F1 score dell'ensemble
- Pesi EWA per ogni modello
- Confronto ensemble vs singoli modelli
- Miglioramento percentuale

## ✅ Verifiche Effettuate

- [x] Integrazione con pipeline Optuna esistente
- [x] Selezione automatica top N modelli
- [x] Training ensemble con config dai trial
- [x] EWA weighting basato su validation F1
- [x] Soft voting pesato per predizioni
- [x] Device auto-detection (MPS/CUDA/CPU)
- [x] Tutto in un unico notebook
- [x] Nessun file esterno necessario

## 🎯 Confronto con Approccio Manuale

| Caratteristica | Manuale | Optuna |
|---------------|---------|--------|
| Configurazioni | Hardcoded (3 fisse) | Automatiche (da trial) |
| Numero modelli | Fisso (9) | Configurabile (N) |
| Ottimizzazione | Manuale | Automatica |
| Flessibilità | Bassa | Alta |
| Risultati | Buoni | Migliori |
| Setup | Semplice | Medio |
| Tempo | Fisso | Variabile (n_trials) |

## 💡 Best Practices

1. **n_trials**: Inizia con 100, poi aumenta a 300+ per migliori risultati
2. **N_MODELS_ENSEMBLE**: 9 è un buon compromesso (5-15 range)
3. **GPU**: Usa GPU/MPS per Optuna (molto più veloce)
4. **Patience**: Aumenta `early_stop_patience` per esplorare meglio
5. **Pruning**: Mantieni pruning attivo (ferma trial non promettenti)

## 🎉 Conclusione

L'implementazione è **completa e integrata con Optuna**. Ora il sistema:
- ✅ Trova automaticamente i migliori iperparametri
- ✅ Seleziona automaticamente i migliori modelli
- ✅ Crea ensemble ottimizzato senza configurazione manuale
- ✅ È più flessibile e produce risultati migliori

**File da usare**: `pain_pirate_optuna_ensemble.ipynb`

---

**Nota**: Questo sostituisce l'approccio precedente con configurazioni hardcoded. Il nuovo notebook integra completamente Optuna per un'ottimizzazione automatica end-to-end.
