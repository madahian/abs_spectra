import numpy as np
import os
import sys
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from .utils import mean_absolute_error_ev, wavelength_to_ev, bitstring_to_array

import matplotlib.pyplot as plt
import seaborn as sns

import config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def prepare_features_targets(train_df, test_df):
    """
    Prepare features (Morgan fingerprints) and targets (primary wavelength) for training.
    """
    # Convert fingerprints from bitstring to numerical arrays
    X_train = np.vstack(train_df["MorganFingerprint"].apply(bitstring_to_array).values)
    X_test = np.vstack(test_df["MorganFingerprint"].apply(bitstring_to_array).values)

    # Use primary wavelength as target
    y_train = train_df["PrimaryWavelength"].values
    y_test = test_df["PrimaryWavelength"].values

    logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_and_tune_rf(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [134],
        "max_depth": [78],
        "max_features": [0.3007],
        "min_samples_split": [2],
        "bootstrap": [True],
    }
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Wavelength metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Energy metrics
    mae_ev = mean_absolute_error_ev(y_test, y_pred)

    logging.info(f"Test MSE (wavelength): {mse:.4f} nm²")
    logging.info(f"Test R²: {r2:.4f}")
    logging.info(f"Test MAE (energy): {mae_ev:.4f} eV")

    return y_pred, mse, r2, mae_ev


def plot_results(y_test, y_pred):
    # Wavelength plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    ax1.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
    )
    ax1.set_xlabel("Actual Wavelength (nm)")
    ax1.set_ylabel("Predicted Wavelength (nm)")
    ax1.set_title("Predicted vs. Actual Wavelength")

    # Energy plot
    y_test_ev = wavelength_to_ev(y_test)
    y_pred_ev = wavelength_to_ev(y_pred)

    sns.scatterplot(x=y_test_ev, y=y_pred_ev, ax=ax2)
    ax2.plot(
        [min(y_test_ev), max(y_test_ev)],
        [min(y_test_ev), max(y_test_ev)],
        color="red",
        linestyle="--",
    )
    ax2.set_xlabel("Actual Energy (eV)")
    ax2.set_ylabel("Predicted Energy (eV)")
    ax2.set_title("Predicted vs. Actual Energy")

    plt.tight_layout()
    plt.show()

    # Error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Wavelength errors
    errors_nm = y_pred - y_test
    sns.histplot(errors_nm, kde=True, ax=ax1)
    ax1.axvline(0, color="red", linestyle="--")
    ax1.set_xlabel("Wavelength Error (nm)")
    ax1.set_ylabel("Count")
    ax1.set_title("Wavelength Error Distribution")

    # Energy errors
    errors_ev = y_pred_ev - y_test_ev
    sns.histplot(errors_ev, kde=True, ax=ax2)
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Energy Error (eV)")
    ax2.set_ylabel("Count")
    ax2.set_title("Energy Error Distribution")

    plt.tight_layout()
    plt.show()


def main():
    train_csv = config.TRAIN_DATA_CSV
    test_csv = config.TEST_DATA_CSV

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train, X_test, y_train, y_test = prepare_features_targets(train_df, test_df)
    best_model = train_and_tune_rf(X_train, y_train)
    y_pred, mse, r2, mae_ev = evaluate_model(best_model, X_test, y_test)

    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()
