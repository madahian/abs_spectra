# Digital Alchemy: Molecular Absorption Wavelength Prediction

A Random Forest regression model for predicting absorption wavelengths corresponding to a maximum in the spectrum of molecules using their structural information (SMILES representation).

## Project Overview

This project implements a machine learning pipeline to predict molecular absorption wavelength corresponding to a maximum in the spectrum using Random Forest regression. It processes molecular spectral data, extracts features using Morgan fingerprints, and trains a model to predict the primary absorption wavelength.

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
- Extracts primary absorption peak
- Retrieves SMILES representations from PubChem using batch processing
- Implements parallel processing for file handling
- Performs outlier detection using IQR method for wavelength and absorption values
- Creates processed molecular datasets with:
  - Molecule identifiers (CAS, Name)
  - SMILES representation
  - Primary absorption wavelength
  - Outlier flags for wavelength and absorption values

### 2. Model Training (`src/model_training.py`)

- Converts SMILES to canonical form
- Generates 1024-bit Morgan fingerprints
- Extracts primary absorption peak
- Implements stratified train/test splitting by molecule (CAS number)
  - Ensures balanced distribution of outliers between splits
  - Prevents data leakage across splits

## Model Evaluation (`notebooks/rf_model_pipeline.ipynb`)

### Features

- Morgan fingerprints (1024 bits)
- Primary wavelength target

### Model Architecture

- Random Forest Regressor with optimized hyperparameters:
  - n_estimators: 134
  - max_depth: 78
  - max_features: 0.3007
  - min_samples_split: 2
  - bootstrap: True

### Hyperparameter Optimization

The model supports multiple optimization strategies:
1. **Default Mode**: Uses pre-optimized parameters for stable performance
2. **Broad Search**: RandomizedSearchCV for extensive parameter space exploration
3. **Advanced Optimization**: Optuna integration for sophisticated hyperparameter tuning

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

### Optimization Methods

To use different optimization strategies:

```python
# For stable use (pre-optimized parameters)
model = train_and_tune_rf(X_train, y_train, optimization_method='final')

# For broad parameter space exploration
model = train_and_tune_rf(X_train, y_train, optimization_method='broad_search')

# For Optuna optimization
model = train_and_tune_rf(X_train, y_train, optimization_method='optuna')
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
- optuna
- Jupyter Notebook

## License

See the [LICENSE](LICENSE) file for details.
