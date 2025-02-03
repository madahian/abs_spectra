# Digital Alchemy: Molecular Absorption Wavelength Prediction

A Random Forest regression model for predicting absorption wavelengths corresponding to a maximum in the spectrum of molecules using their structural information (SMILES representation).

## Project Overview

This project implements a machine learning pipeline to predict molecular absorption wavelengths corresponding to a maximum in the spectrum using Random Forest regression. It processes molecular spectral data, extracts features using Morgan fingerprints, and trains a model to predict primary and secondary absorption wavelengths.

## File Structure

```
.
├── data/
│   ├── data_checker.py         # Data validation and visualization script
│   ├── molecular_data.csv      # Raw molecular data
│   ├── training_data.csv       # Processed training dataset
│   └── test_data.csv           # Processed test dataset
├── src/
│   ├── data_prep.ipynb         # Data preparation notebook
│   ├── generate_train_test.py  # Train/test split generation script
│   └── train_rf_model.ipynb    # Model training and evaluation notebook
└── PhotochemCAD/               # Raw spectral data directory
```

## Data Processing Pipeline

### 1. Data Preparation (`data_prep.ipynb`)

- Processes raw absorption spectra from `.abs.txt` files
- Extracts up to 3 highest absorption peaks
- Retrieves SMILES representations from PubChem
- Creates initial molecular dataset with:
  - Molecule identifiers (Code, CAS, Name)
  - SMILES representation
  - Absorption maxima and wavelengths

### 2. Data Validation (`data_checker.py`)

- Checks for missing values
- Visualizes wavelength and absorption maxima distributions
- Helps identify potential data quality issues

### 3. Train/Test Generation (`generate_train_test.py`)

- Converts SMILES to canonical form
- Generates 1024-bit Morgan fingerprints
- Extracts primary and secondary absorption peaks
- Splits data into training and test sets by molecule (CAS number)
- Ensures no data leakage between splits

## Model Training (`train_rf_model.ipynb`)

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

Before running the data preparation notebook, you need to download and set up the PhotochemCAD dataset from [https://photochemcad.com/download](https://photochemcad.com/download):

1. Download the PhotochemCAD dataset
2. After extraction, place the dataset in the root directory of the project as 'PhotochemCAD' folder
3. Ensure the 'PhotochemCAD' folder contains the 'Common Compounds' folder which includes `.abs.txt` files

### Data Processing and Model Training

1. Data Preparation:

```bash
# Process raw spectral data and create molecular dataset
jupyter notebook src/data_prep.ipynb
```

2. Generate Train/Test Sets:

```bash
# Create training and test datasets
python src/generate_train_test.py
```

3. Train Model:

```bash
# Train and evaluate the Random Forest model
jupyter notebook src/train_rf_model.ipynb
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
