import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import randint, uniform
import optuna
from .utils import (
    wavelength_to_ev,
    bitstring_to_array,
    generate_performance_report,
    detect_prediction_outliers
)

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
    return X_train, X_test, y_train, y_test, train_df, test_df

def train_anomaly_detector(X_train):
    """
    Train an isolation forest model for anomaly detection.
    """
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)
    return iso_forest

def objective(trial, X_train, y_train):
    """
    Objective function for Optuna optimization.
    """
    rf = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 300),
        max_depth=trial.suggest_int('max_depth', 50, 100),
        max_features=trial.suggest_float('max_features', 0.1, 0.5),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        random_state=42
    )
    
    score = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return score.mean()

def train_and_tune_rf(X_train, y_train, optimization_method='final'):
    """
    Train and tune Random Forest with multiple optimization strategies.
    
    Args:
        optimization_method: 'broad_search', 'fine_tune', 'optuna', or 'final'
    """
    if optimization_method == 'broad_search':
        rf = RandomForestRegressor(random_state=42)
        param_distributions = {
            "n_estimators": randint(100, 300),
            "max_depth": randint(50, 100),
            "max_features": uniform(0.1, 0.4),
            "min_samples_split": randint(2, 10),
        }
        
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=100,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_
        
    elif optimization_method == 'optuna':
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
        
        best_params = study.best_params
        rf = RandomForestRegressor(**best_params, random_state=42)
        rf.fit(X_train, y_train)
        return rf
        
    else:  # 'final' - use pre-optimized parameters
        # Use the optimal parameters found
        rf = RandomForestRegressor(
            n_estimators=134,
            max_depth=78,
            max_features=0.3007,
            min_samples_split=2,
            bootstrap=True,
            random_state=42
        )
        rf.fit(X_train, y_train)
        logging.info("Using pre-optimized parameters for final model")
        return rf

def plot_cv_results(cv_results):
    """Plot cross-validation results."""
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(1, len(cv_results['mean_test_score']) + 1),
                -cv_results['mean_test_score'],
                yerr=cv_results['std_test_score'])
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-validation Results')
    plt.show()

def evaluate_model(model, X_test, y_test, anomaly_detector=None, test_df=None):
    y_pred = model.predict(X_test)

    # Detect prediction outliers
    pred_outliers = detect_prediction_outliers(y_test, y_pred)
    
    # Detect anomalies using isolation forest if available
    anomalies = None
    if anomaly_detector is not None:
        anomalies = anomaly_detector.predict(X_test) == -1
    
    # Get statistical outliers if available
    stat_outliers = None
    if test_df is not None and "Is_Outlier" in test_df.columns:
        stat_outliers = test_df["Is_Outlier"].values
    
    # Generate performance report
    report = generate_performance_report(y_test, y_pred, pred_outliers, anomalies, stat_outliers)
    print(report)

    return y_pred, pred_outliers, anomalies, stat_outliers

def plot_results(y_test, y_pred, outliers=None):
    # Create color array for outliers
    colors = ['blue' if not out else 'red' for out in (outliers if outliers is not None else np.zeros_like(y_test))]
    
    # Wavelength plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_test, y_pred, c=colors, alpha=0.6)
    ax1.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="black",
        linestyle="--",
    )
    ax1.set_xlabel("Actual Wavelength (nm)")
    ax1.set_ylabel("Predicted Wavelength (nm)")
    ax1.set_title("Predicted vs. Actual Wavelength")
    if outliers is not None:
        ax1.legend(['Perfect Prediction', 'Normal', 'Outlier'])

    # Energy plot
    y_test_ev = wavelength_to_ev(y_test)
    y_pred_ev = wavelength_to_ev(y_pred)

    ax2.scatter(y_test_ev, y_pred_ev, c=colors, alpha=0.6)
    ax2.plot(
        [min(y_test_ev), max(y_test_ev)],
        [min(y_test_ev), max(y_test_ev)],
        color="black",
        linestyle="--",
    )
    ax2.set_xlabel("Actual Energy (eV)")
    ax2.set_ylabel("Predicted Energy (eV)")
    ax2.set_title("Predicted vs. Actual Energy")
    if outliers is not None:
        ax2.legend(['Perfect Prediction', 'Normal', 'Outlier'])

    plt.tight_layout()
    plt.show()

    # Error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Wavelength errors
    errors_nm = y_pred - y_test
    if outliers is not None:
        sns.histplot(data=pd.DataFrame({
            'Error': errors_nm,
            'Type': ['Outlier' if out else 'Normal' for out in outliers]
        }), x='Error', hue='Type', multiple="stack", ax=ax1)
    else:
        sns.histplot(errors_nm, kde=True, ax=ax1)
    ax1.axvline(0, color="black", linestyle="--")
    ax1.set_xlabel("Wavelength Error (nm)")
    ax1.set_ylabel("Count")
    ax1.set_title("Wavelength Error Distribution")

    # Energy errors
    errors_ev = y_pred_ev - y_test_ev
    if outliers is not None:
        sns.histplot(data=pd.DataFrame({
            'Error': errors_ev,
            'Type': ['Outlier' if out else 'Normal' for out in outliers]
        }), x='Error', hue='Type', multiple="stack", ax=ax2)
    else:
        sns.histplot(errors_ev, kde=True, ax=ax2)
    ax2.axvline(0, color="black", linestyle="--")
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

    X_train, X_test, y_train, y_test, train_df, test_df = prepare_features_targets(train_df, test_df)

    # Train anomaly detector
    anomaly_detector = train_anomaly_detector(X_train)

    # Train main model with optimization methods
    best_model = train_and_tune_rf(X_train, y_train, optimization_method="final")

    # Evaluate with outlier detection and generate report
    y_pred, pred_outliers, anomalies, stat_outliers = evaluate_model(
        best_model, X_test, y_test, 
        anomaly_detector=anomaly_detector,
        test_df=test_df
    )

    plot_results(y_test, y_pred, pred_outliers)

if __name__ == "__main__":
    main()
