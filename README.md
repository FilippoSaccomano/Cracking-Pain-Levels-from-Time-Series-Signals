# challenge_an2dl
Challenge AN2DL - Pain Classification

## 🎯 Ensemble con Optuna (RECOMMENDED)

Per ottenere i migliori risultati, usa l'ensemble automatico con Optuna:

**File**: `pain_pirate_optuna_ensemble.ipynb`

Questo notebook contiene:
- ✅ **Optuna optimization** per trovare automaticamente i migliori iperparametri
- ✅ **Selezione automatica** dei migliori N modelli dai trial di Optuna
- ✅ **Ensemble** dei modelli selezionati con EWA weighting
- ✅ Pipeline completamente automatizzata in un unico file

### Quick Start
1. Assicurati di avere i file dati: `pirate_pain_train.csv`, `pirate_pain_train_labels.csv`, `pirate_pain_test.csv`
2. Apri `pain_pirate_optuna_ensemble.ipynb`
3. Esegui tutte le celle (`Run All`)
4. Ottieni `submission_optuna_ensemble.csv`

### Vantaggi
- 🔍 Optuna trova automaticamente i migliori iperparametri (es. 300 trial)
- 🎯 Ensemble usa automaticamente i top N modelli migliori
- ⚖️ EWA pesa i modelli basandosi sulle performance
- 🚀 Non servono configurazioni manuali

Vedi `README_ENSEMBLE.md` per dettagli completi.

## 📁 Altri File

- `pain_pirate_complete_pytorch.ipynb` - Singolo modello baseline
- `pain_pirate_K_Fold.ipynb` - Approccio K-Fold
- `best_model_pytorch.pth` - Best single model weights
- `best_config.json` - Best single model configuration
