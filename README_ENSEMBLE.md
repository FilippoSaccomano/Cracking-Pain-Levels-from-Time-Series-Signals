# 🏴‍☠️ Ensemble di 9 Modelli Migliori

## File Principale

**`pain_pirate_ensemble_complete.ipynb`** - Notebook completo con ensemble di 9 modelli

## Cosa Fa Questo Notebook

Questo notebook implementa un **ensemble di 9 modelli Conv1D + BiLSTM** per migliorare le predizioni:

### Architettura
- **3 configurazioni** di iperparametri
- **3 seeds** diversi per ogni configurazione
- **Totale: 9 modelli**

### Configurazioni
1. **128x128x128**: conv_dim=128, lstm_units=128
2. **128x128x160**: conv_dim=128, lstm_units=160  
3. **160x160x160**: conv_dim=160, lstm_units=160

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
1. Apri `pain_pirate_ensemble_complete.ipynb`
2. Esegui tutte le celle in ordine (`Run All`)
3. Il notebook genererà `submission_ensemble_9models.csv`

### 4. Struttura del Notebook

```
1. Caricamento dati
2. Preprocessing (autocorrelation, features categoriche, time features)
3. Creazione finestre sliding
4. Split train/validation
5. ⭐ Definizione architettura Conv1D + BiLSTM + funzioni ensemble
6. ⭐ Training 9 modelli
7. ⭐ Valutazione e calcolo pesi EWA
8. ⭐ Predizione test con ensemble
9. Creazione submission
10. Visualizzazione risultati
```

## Output

### File Generato
- **`submission_ensemble_9models.csv`**: Predizioni ensemble per il test set

### Metriche Visualizzate
- F1 score individuale per ogni modello
- F1 score dell'ensemble
- Pesi EWA per ogni modello
- Distribuzione delle predizioni
- Grafici di confronto

## Vantaggi dell'Ensemble

✅ **Riduzione varianza**: Media di 9 modelli riduce overfitting  
✅ **Migliore generalizzazione**: Seed diversi esplorano diversi minimi locali  
✅ **Voting pesato**: I modelli migliori hanno più influenza  
✅ **Robustezza**: Meno sensibile a singoli modelli fallimentari  

## Tempi di Esecuzione

| Hardware | Tempo per modello | Tempo totale |
|----------|------------------|--------------|
| CPU      | 20-40 min        | 3-6 ore      |
| GPU/CUDA | 5-10 min         | 45-90 min    |
| MPS (M1/M2/M3/M4) | 5-10 min  | 45-90 min    |

## Note Tecniche

- Il notebook rileva automaticamente il device disponibile (MPS/CUDA/CPU)
- Ogni modello usa early stopping basato su validation F1
- Label smoothing e dropout per regularizzazione
- Batch normalization per stabilità
- Gradient clipping per prevenire esplosione gradienti

## Troubleshooting

**Problema**: CUDA out of memory  
**Soluzione**: Modifica `batch_size` da 16 a 8 nella funzione `train_one_model`

**Problema**: Training troppo lento  
**Soluzione**: Riduci `max_epochs` nelle configurazioni o usa GPU

**Problema**: F1 dell'ensemble peggiore dei singoli modelli  
**Soluzione**: Verifica che almeno 7-8 modelli su 9 abbiano buoni F1 score

## File NON Necessari

❌ `ensemble_pytorch.py` - ELIMINATO (tutto è nel notebook)  
❌ File esterni Python - NON servono  
✅ Solo `pain_pirate_ensemble_complete.ipynb` è necessario
