# EEG-Based Stroke Detection using Machine Learning

## Overview
This project implements a machine learning system for detecting stroke patients using EEG (Electroencephalogram) data. The system processes raw EEG signals, extracts relevant features, and uses a Random Forest classifier with SMOTE balancing to classify between stroke and normal patients.

## Key Features
- EEG signal processing and feature extraction
- Advanced feature engineering from multiple EEG channels
- SMOTE-based class balancing
- Optimized Random Forest classification
- Comprehensive data preprocessing pipeline
- 
```


### Feature Categories
1. Time Domain Features:
   - Mean, Variance, RMS
   - Line Length
   - Peak-to-Peak Amplitude
   - Zero Crossing Rate
   - Hjorth Parameters

2. Frequency Domain Features:
   - Band Powers (Delta, Theta, Alpha, Beta)
   - Spectral Entropy
   - Band Power Ratios

3. Non-linear Features:
   - Approximate Entropy
   - Sample Entropy
   - Wavelet Energy
   - Wavelet Coefficients

### Model Architecture
- Pipeline: SMOTE → StandardScaler → RandomForestClassifier
- Cross-validation: 8-fold
- Hyperparameter optimization using GridSearchCV

 Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- MNE
- PyWavelets
- imbalanced-learn
- SciPy
- Matplotlib
- Seaborn

## Usage

1. Data Preprocessing:
```python
python src/preprocess.py
```

2. Feature Extraction:
```python
python src/feature_extraction.py
```

3. Model Training:
```python
python src/train_model.py
```
