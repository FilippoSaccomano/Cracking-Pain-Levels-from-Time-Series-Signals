# challenge_an2dl
Challenge AN2DL - Pain Classification

## 🎯 Ensemble Approach (RECOMMENDED)

Per ottenere i migliori risultati, usa l'ensemble di 9 modelli:

**File**: `pain_pirate_ensemble_complete.ipynb`

Questo notebook contiene:
- ✅ Tutto il codice in un unico file
- ✅ Ensemble di 9 modelli Conv1D + BiLSTM
- ✅ EWA weighting basato su validation F1
- ✅ Predizioni con soft voting pesato

### Quick Start
1. Assicurati di avere i file dati: `pirate_pain_train.csv`, `pirate_pain_train_labels.csv`, `pirate_pain_test.csv`
2. Apri `pain_pirate_ensemble_complete.ipynb`
3. Esegui tutte le celle (`Run All`)
4. Ottieni `submission_ensemble_9models.csv`

Vedi `README_ENSEMBLE.md` per dettagli completi.

## 📁 Altri File

- `pain_pirate_complete_pytorch.ipynb` - Singolo modello baseline
- `pain_pirate_K_Fold.ipynb` - Approccio K-Fold
- `best_model_pytorch.pth` - Best single model weights
- `best_config.json` - Best single model configuration
