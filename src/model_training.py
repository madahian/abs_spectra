import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def bitstring_to_array(bitstring):
    """Convert a fingerprint bitstring to a numpy array."""
    return np.array([int(bit) for bit in bitstring])


def prepare_features_targets(train_df, test_df):
    # Convert fingerprints from bitstring to numerical arrays
    X_train = np.vstack(train_df["MorganFingerprint"].apply(bitstring_to_array).values)
    X_test = np.vstack(test_df["MorganFingerprint"].apply(bitstring_to_array).values)

    # Create binary flags for secondary peaks
    for df in [train_df, test_df]:
        df["Has_Secondary1"] = df["SecondaryWavelength1"].notna().astype(int)
        df["Has_Secondary2"] = df["SecondaryWavelength2"].notna().astype(int)

    X_train = np.hstack(
        [X_train, train_df[["Has_Secondary1", "Has_Secondary2"]].values]
    )
    X_test = np.hstack([X_test, test_df[["Has_Secondary1", "Has_Secondary2"]].values])

    # Fill missing secondary wavelengths with 0 and create weighted target using config.PEAK_WEIGHTS
    for df in [train_df, test_df]:
        df["SecondaryWavelength1"] = df["SecondaryWavelength1"].fillna(0)
        df["SecondaryWavelength2"] = df["SecondaryWavelength2"].fillna(0)
        df["WeightedWavelength"] = (
            config.PEAK_WEIGHTS["primary"] * df["PrimaryWavelength"]
            + config.PEAK_WEIGHTS["secondary1"] * df["SecondaryWavelength1"]
            + config.PEAK_WEIGHTS["secondary2"] * df["SecondaryWavelength2"]
        )

    y_train = train_df["WeightedWavelength"].values
    y_test = test_df["WeightedWavelength"].values

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
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Test MSE: {mse:.4f}")
    logging.info(f"Test R^2: {r2:.4f}")
    return y_pred, mse, r2


def plot_results(y_test, y_pred):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Actual Wavelength")
    plt.ylabel("Predicted Wavelength")
    plt.title("Predicted vs. Actual Wavelength")
    plt.show()

    errors = y_pred - y_test
    plt.figure(figsize=(6, 4))
    sns.histplot(errors, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Prediction Error (Residuals)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.show()


def main():
    train_csv = config.TRAIN_DATA_CSV
    test_csv = config.TEST_DATA_CSV

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train, X_test, y_train, y_test = prepare_features_targets(train_df, test_df)
    best_model = train_and_tune_rf(X_train, y_train)
    y_pred, mse, r2 = evaluate_model(best_model, X_test, y_test)
    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()
