# Digital Alchemy: Molecular Absorption Wavelength Prediction

A Random Forest regression model for predicting absorption wavelengths corresponding to a maximum in the spectrum of molecules using their structural information (SMILES representation).

## Project Overview

This project implements a machine learning pipeline to predict molecular absorption wavelengths corresponding to a maximum in the spectrum using Random Forest regression. It processes molecular spectral data, extracts features using Morgan fingerprints, and trains a model to predict primary and secondary absorption wavelengths.

## File Structure

```
.
├── config.py                # Configuration parameters and settings
├── data/
│   ├── processed/         # Processed datasets directory
│   ├── raw/               # Raw data directory
│   │   └── PhotochemCAD/  # Raw spectral data
├── notebooks/
│   └── rf_model_pipeline.ipynb  # Model training and evaluation notebook
└── src/
    ├── data_processing.py     # Data processing and feature engineering
    ├── model_training.py      # Model training and hyperparameter optimization
    └── utils.py               # Utility functions and helpers
```

## Data Processing Pipeline

### 1. Data Processing (`src/data_processing.py`)

- Processes raw absorption spectra from `.abs.txt` files
- Extracts up to 3 highest absorption peaks
- Retrieves SMILES representations from PubChem
- Creates processed molecular datasets with:
  - Molecule identifiers (CAS, Name)
  - SMILES representation
  - Absorption maxima and wavelengths

### 2. Model Training (`src/model_training.py`)

- Converts SMILES to canonical form
- Generates 1024-bit Morgan fingerprints
- Extracts primary and secondary absorption peaks
- Implements train/test splitting by molecule (CAS number)
- Ensures no data leakage between splits

## Model Evaluation (`notebooks/rf_model_pipeline.ipynb`)

### Features

- Morgan fingerprints (1024 bits)
- Secondary peak presence flags
- Weighted wavelength targets

### Model Architecture

- Random Forest Regressor
- Hyperparameter optimization via GridSearch/BayesSearchCV
- 5-fold cross-validation

### Performance Metrics

- Mean Squared Error (MSE)
- R-squared (R²)
- Visualization of predictions vs. actuals
- Error distribution analysis

## Usage

### Dataset Setup

Before running the data processing script, you need to download and set up the PhotochemCAD dataset from [https://photochemcad.com/download](https://photochemcad.com/download):

1. Download the PhotochemCAD dataset
2. After extraction, place the dataset in the data/raw directory as 'PhotochemCAD' folder
3. Ensure the 'PhotochemCAD' folder contains the 'Common Compounds' folder which includes `.abs.txt` files

### Data Processing and Model Training

1. Via jupyter notebook:

```bash
# Run the whole pipeline
jupyter notebook notebooks/rf_model_pipeline.ipynb
```

2. Alternatively:

Data Processing:

```bash
# Process raw spectral data and create molecular dataset
python -m src.data_processing
```

Train Model:

```bash
# Train and evaluate the Random Forest model
python -m src.model_training
```

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- RDKit
- matplotlib
- seaborn
- scikit-optimize
- PubChemPy
- Jupyter Notebook

## License

See the [LICENSE](LICENSE) file for details.
