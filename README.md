# EEG-GAN: Synthetic EEG Data Generation for Enhanced Motor Imagery Classification
![EEG Sample](stuff/eeg.jpg)

## 🧠 Overview

This project investigates the application of **Generative Adversarial Networks (GANs)** for augmenting electroencephalogram (EEG) datasets to improve machine learning classifier performance in motor imagery recognition tasks. By generating realistic synthetic EEG signals, we address the critical challenge of data scarcity in brain-computer interface (BCI) development.

## 🎯 Key Features

- **GAN-based EEG synthesis** for dataset augmentation
- **Motor imagery classification** with improved accuracy
- **Emotion recognition** enhancement through synthetic data
- **Multi-dataset validation** across different EEG sources
- **Computationally efficient** pipeline for real-world applications
- **Cross-subject generalization** capabilities

## 📊 Results Highlights

- ✅ Improved classification accuracy on motor imagery tasks
- ✅ Enhanced emotion recognition performance 
- ✅ Realistic synthetic EEG signals preserving temporal characteristics
- ✅ Reduced training time through efficient augmentation pipeline
- ✅ Demonstrated cross-subject generalization improvements

## 🗂️ Datasets Used

### Primary Datasets
1. **DEAP Database** - Database for Emotion Analysis using Physiological Signals
   - [Dataset Link](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
   - Multi-channel EEG recordings for emotion recognition

2. **Kaggle EEG Collection** - Comprehensive EEG dataset
   - [Dataset Link](https://www.kaggle.com/datasets/jbouv27/eeg)
   - Various EEG signal types and classifications

3. **Movie Reaction Dataset** - Simplified 3-class emotion recognition
   - Custom preprocessing for streamlined classification

## 🏗️ Architecture

### GAN Components
- **Generator Network**: Creates synthetic EEG signals from noise input
- **Discriminator Network**: Distinguishes between real and synthetic signals
- **Training Pipeline**: Adversarial training with stability enhancements

### Data Processing Pipeline
```
Raw EEG Data → Preprocessing → Feature Extraction → GAN Training → Synthetic Data → Enhanced Classification
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow keras numpy scipy matplotlib scikit-learn pandas
```

### Installation
```bash
git clone https://github.com/yourusername/eeg-gan-synthesis.git
cd eeg-gan-synthesis
pip install -r requirements.txt
```

### Usage
```python
# Load and preprocess data
python data_preprocessing.py

# Train the GAN model
python train_gan.py --epochs 100 --batch_size 32

# Generate synthetic EEG data
python generate_synthetic.py --num_samples 1000

# Evaluate classification performance
python evaluate_classifier.py --use_synthetic True
```

## 📁 Project Structure

```
eeg-gan-synthesis/
├── data/
│   ├── raw/                 # Raw EEG datasets
│   ├── processed/           # Preprocessed data
│   └── synthetic/           # Generated synthetic data
├── models/
│   ├── gan_model.py         # GAN architecture
│   ├── classifier.py        # Classification models
│   └── preprocessing.py     # Data preprocessing utilities
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── scripts/
│   ├── train_gan.py
│   ├── generate_synthetic.py
│   └── evaluate_classifier.py
├── results/
│   ├── figures/             # Generated plots and visualizations
│   └── metrics/             # Performance metrics
└── README.md
```

## 📈 Performance Metrics

### Motor Imagery Classification
- **Baseline Accuracy**: X.X%
- **With GAN Augmentation**: X.X%
- **Improvement**: +X.X%

### Emotion Recognition
- **3-Class Classification**: X.X% accuracy
- **Cross-Subject Validation**: X.X% accuracy
- **Synthetic Data Quality**: High spectral similarity

## 🔬 Technical Approach

### Data Preprocessing
1. **Noise Removal**: Advanced filtering techniques
2. **Signal Processing**: Band-pass filtering and normalization
3. **Feature Extraction**: Time-domain, frequency-domain, and time-frequency features
4. **Segmentation**: Optimal window sizing with overlap strategies

### GAN Training Strategy
- **Loss Functions**: Adversarial loss with gradient penalty
- **Optimization**: Adam optimizer with learning rate scheduling
- **Stability Techniques**: Progressive growing and spectral normalization
- **Convergence Monitoring**: Real-time loss tracking and quality metrics

## 🎓 Research Applications

- **Brain-Computer Interfaces (BCIs)**
- **Neurological disorder diagnosis**
- **Cognitive state monitoring**
- **Motor imagery rehabilitation systems**
- **Emotion recognition systems**

## 📚 Publications & References

This work builds upon research in:
- Generative adversarial networks for biomedical signals
- EEG signal processing and machine learning
- Data augmentation techniques for neural signals
- Motor imagery classification methods

## 👥 Team & Collaboration

**Project Duration**: 5 days  
**Team Size**: 6 members  
**Development Framework**: Collaborative research implementation

