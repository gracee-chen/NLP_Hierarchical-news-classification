# Hierarchical News Classification

<p align="center">
<img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" alt="Framework">
<img src="https://img.shields.io/badge/Model-BERT-yellow.svg" alt="Model">
</p>

## A Two-Level Classification System for News Articles

[Grace Chen<sup>1</sup>](mailto:gc3022@nyu.edu), [Zoey Qu<sup>1</sup>](mailto:xq529@nyu.edu), [Michael Zhao<sup>1</sup>](mailto:bz2227@nyu.edu)

<sup>1</sup> New York University

## Overview

This project implements a hierarchical news classification system that leverages BERT embeddings with a two-level cascade architecture. The system first classifies news articles into broad categories (Level 1) and then further classifies them into specific subcategories (Level 2). By addressing hierarchical classification challenges through innovative confidence management and error correction strategies, our approach provides more accurate and robust news categorization, enhancing both computational efficiency and practical interpretability.

## Key Innovations

- **Adaptive Confidence Thresholds**: Dynamically adjusts confidence thresholds based on historical accuracy to mitigate error propagation
- **Data Balancing Strategy**: Implements intelligent upsampling for under-represented subcategories to improve classification of rare classes
- **Hierarchical Error Correction**: Introduces a feedback mechanism between classification levels, allowing low-confidence primary classifications to be corrected
- **Differentiated Training Parameters**: Optimizes hyperparameters separately for each classification level

## Model Architecture


## Files Structure

- `config.py`: Configuration parameters for models and training
- `data.py`: Data loading, preprocessing, and balancing functions
- `thresholds.py`: Implementation of adaptive threshold mechanism
- `train_l1.py`: Training script for Level-1 classifier
- `train_l2.py`: Training script for Level-2 classifiers
- `predict.py`: Model inference and hierarchical classification
- `app.py`: Streamlit web application for interactive classification

## Requirements

- Python 3.7+
- PyTorch >= 1.9.0
- Transformers >= 4.12.0
- Pandas >= 1.3.0
- NumPy >= 1.20.0
- Scikit-learn >= 0.24.0
- Datasets >= 1.11.0
- Evaluate >= 0.2.0
- Streamlit >= 1.8.0

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train Level-1 classifier
python train_l1.py

# Train Level-2 classifiers
python train_l2.py

# Run interactive prediction
python predict.py

# Launch the web interface
streamlit run app.py
```

## Evaluation Results


## Web Interface

The Streamlit application provides an interactive interface for classifying news articles and visualizing confidence scores for both levels.

## Citation

