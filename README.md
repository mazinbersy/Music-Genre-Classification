# 🎵 Music Genre Classification

This project classifies music tracks into **10 genres** using two different approaches:

1. **Feature Extraction + XGBoost (Tabular Data)**
2. **Spectrograms + Convolutional Neural Network (Image Data)**
3. 3. **Spectrograms + Transfer Learning (VGG16)**

Dataset used: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (10 genres, 1000 audio files).

---

## 📂 Project Structure
```
├── Data/
│ └── genres_original/ # Original GTZAN audio files (.wav)
├── spectrograms/ # Generated spectrogram images
├── model_tabular.ipynb # Feature extraction + XGBoost
├── model_image.ipynb # Spectrogram + CNN + Transfer Learning
└── README.md
```
---

## ⚙️ Installation
### Install dependencies
```bash
pip install numpy pandas librosa matplotlib scikit-learn xgboost tensorflow
```
---
## 🚀Usage
### 1. Feature Extraction + XGBoost
- Runs MFCCs, chroma, spectral features, and tempo extraction
- Trains an XGBoost model
- Performs RandomizedSearchCV hyperparameter tuning
- Prints accuracy and best parameters

### 2. Spectrograms + CNN
- Generates spectrograms from audio files
- Trains a simple CNN model on spectrograms
- Outputs training/validation accuracy plots and confusion matrix

### 3. Spectrograms + Transfer Learning (VGG16)
- Uses VGG16 pre-trained on ImageNet
- Freezes convolutional base, adds custom dense layers
- Fine-tunes last layers of VGG16 with a lower learning rate
- Achieves higher accuracy than basic CNN
- Outputs accuracy/loss plots and confusion matrix
---
## 📊 Results
- XGBoost (Feature-based): Accuracy ~ 65–75%
- CNN (Spectrogram-based): Accuracy ~ 45–50% with basic CNN
- Transfer Learning (VGG16): Accuracy ~ 50–60% after fine-tuning
