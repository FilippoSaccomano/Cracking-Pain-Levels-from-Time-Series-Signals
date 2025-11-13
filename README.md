# 🏴‍☠️ Challenge AN2DL - Pain Pirate Analysis

Repository per la challenge di Artificial Neural Networks and Deep Learning.

## 📚 File Principali

### Notebook di Analisi

- **`pain_pirate_analysis.ipynb`**: Notebook originale con la pipeline completa
- **`pain_pirate_analysis_enhanced.ipynb`**: ✨ **TensorFlow** - Versione con consigli del professore
- **`pain_pirate_analysis_pytorch.ipynb`**: ✨ **PyTorch** - Versione con consigli del professore  
- **`Timeseries Forecasting.ipynb`**: Notebook di riferimento del corso sulla previsione di serie temporali

💡 **Scegli il framework che preferisci!** Entrambe le versioni enhanced includono tutte le 7 tecniche ADVICE.

### Documentazione

- **`PROFESSOR_ADVICE_GUIDE.md`**: 📖 Guida completa ai consigli del professore e come applicarli

## 🎓 Consigli del Professore Integrati

Il notebook `pain_pirate_analysis_enhanced.ipynb` include tutte le tecniche suggerite dal professore:

1. **📊 ADVICE 11/11**: Analisi di autocorrelazione per scegliere la WINDOW_SIZE ottimale
2. **🕐 ADVICE 12/11**: Time Feature Engineering con encoding ciclico
3. **🔄 ADVICE 13/11**: Architettura ibrida Conv1D + LSTM/GRU
4. **🎯 ADVICE 10/11**: Gradient Clipping per stabilizzare il training
5. **🎲 ADVICE 09/11**: Label Smoothing per ridurre overfitting
6. **⚖️ ADVICE 08/11**: Class Weighting per gestire lo sbilanciamento (già presente)
7. **🗺️ ADVICE 07/11**: Embeddings per features categoriche

## 🚀 Come Iniziare

1. **Scegli il framework**: TensorFlow/Keras o PyTorch
2. **Leggi la guida**: Apri `PROFESSOR_ADVICE_GUIDE.md` per capire ogni tecnica
3. **Esplora il notebook enhanced**: 
   - TensorFlow: `pain_pirate_analysis_enhanced.ipynb`
   - PyTorch: `pain_pirate_analysis_pytorch.ipynb`
4. **Sperimenta**: Prova le tecniche suggerite e confronta i risultati
5. **Monitora F1-score macro**: Usa questa metrica per valutare i miglioramenti

### 🔧 Framework Comparison

| Aspetto | TensorFlow/Keras | PyTorch |
|---------|------------------|---------|
| **API Level** | High-level (Keras) | Mid-level |
| **Training** | `model.fit()` | Custom loops |
| **Learning Curve** | Più facile per iniziare | Più controllo |
| **Industry** | Google, produzione | Research, startup |
| **Best For** | Deployment rapido | Sperimentazione |

**Entrambe le versioni implementano le stesse 7 tecniche ADVICE!**

## 📊 Dataset

- `pirate_pain_train.csv`: Dataset di training con sensori IMU
- Features: accelerometro (acc_x, acc_y, acc_z) e giroscopio (gyro_x, gyro_y, gyro_z)
- Task: Classificazione multiclasse di movimenti/attività

## 🛠️ Setup

### TensorFlow Version
```bash
# Installa le dipendenze
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn statsmodels

# Apri il notebook enhanced
jupyter notebook pain_pirate_analysis_enhanced.ipynb
```

### PyTorch Version
```bash
# Installa le dipendenze
pip install torch numpy pandas matplotlib seaborn scikit-learn statsmodels tqdm

# Apri il notebook PyTorch
jupyter notebook pain_pirate_analysis_pytorch.ipynb
```

## 🎯 Obiettivo della Challenge

Costruire un modello che classifichi accuratamente le attività basandosi sui dati dei sensori IMU, ottenendo il miglior F1-score macro possibile.

---

**Buona fortuna con la challenge! 🎓✨**
