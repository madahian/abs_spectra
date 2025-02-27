import time
import logging
import requests
import numpy as np
from scipy.signal import find_peaks
from pubchempy import NotFoundError, PubChemHTTPError
from sklearn.metrics import mean_squared_error, r2_score
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import config

# Cache for PubChem API calls
smiles_cache = {}
smiles_cache_lock = Lock()

logger = logging.getLogger()


def generate_performance_report(
    y_test, y_pred, outliers, anomalies=None, stat_outliers=None
):
    """
    Generate a comprehensive performance report including outlier analysis.
    """
    report = []
    report.append("=" * 50)
    report.append("PERFORMANCE REPORT")
    report.append("=" * 50)

    # Overall Performance
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_r2 = r2_score(y_test, y_pred)
    overall_mae_ev = mean_absolute_error_ev(y_test, y_pred)

    report.append("\nOVERALL METRICS")
    report.append(f"MSE (wavelength): {overall_mse:.4f} nm²")
    report.append(f"R²: {overall_r2:.4f}")
    report.append(f"MAE (energy): {overall_mae_ev:.4f} eV")

    # Performance excluding prediction outliers
    non_outliers = ~outliers
    clean_mse = mean_squared_error(y_test[non_outliers], y_pred[non_outliers])
    clean_r2 = r2_score(y_test[non_outliers], y_pred[non_outliers])
    clean_mae_ev = mean_absolute_error_ev(y_test[non_outliers], y_pred[non_outliers])

    report.append("\nMETRICS (EXCLUDING PREDICTION OUTLIERS)")
    report.append(f"MSE (wavelength): {clean_mse:.4f} nm²")
    report.append(f"R²: {clean_r2:.4f}")
    report.append(f"MAE (energy): {clean_mae_ev:.4f} eV")

    # Outlier Analysis
    report.append("\nOUTLIER ANALYSIS")
    report.append(f"Total samples: {len(y_test)}")
    report.append(
        f"Prediction outliers: {sum(outliers)} ({sum(outliers)/len(outliers)*100:.1f}%)"
    )

    if anomalies is not None:
        report.append(
            f"Anomalies detected: {sum(anomalies)} ({sum(anomalies)/len(anomalies)*100:.1f}%)"
        )
        report.append(f"Overlap (pred/anom): {sum(outliers & anomalies)} samples")

    if stat_outliers is not None:
        report.append(
            f"Statistical outliers: {sum(stat_outliers)} ({sum(stat_outliers)/len(stat_outliers)*100:.1f}%)"
        )
        report.append(f"Overlap (pred/stat): {sum(outliers & stat_outliers)} samples")
        if anomalies is not None:
            report.append(
                f"Overlap (anom/stat): {sum(anomalies & stat_outliers)} samples"
            )
            report.append(
                f"Agreement across all methods: {sum(outliers & anomalies & stat_outliers)} samples"
            )

    # Error Analysis
    errors = np.abs(y_pred - y_test)
    report.append("\nERROR ANALYSIS")
    report.append(f"Mean absolute error: {np.mean(errors):.4f} nm")
    report.append(f"Error std dev: {np.std(errors):.4f} nm")
    report.append(f"Max error: {np.max(errors):.4f} nm")
    report.append(f"Median error: {np.median(errors):.4f} nm")

    # Outlier-specific metrics
    if sum(outliers) > 0:
        out_mse = mean_squared_error(y_test[outliers], y_pred[outliers])
        out_r2 = r2_score(y_test[outliers], y_pred[outliers])
        out_mae_ev = mean_absolute_error_ev(y_test[outliers], y_pred[outliers])

        report.append("\nOUTLIER METRICS")
        report.append(f"MSE (wavelength): {out_mse:.4f} nm²")
        report.append(f"R²: {out_r2:.4f}")
        report.append(f"MAE (energy): {out_mae_ev:.4f} eV")

    report.append("=" * 50)
    return "\n".join(report)


def detect_prediction_outliers(y_true, y_pred, threshold=2):
    """
    Detect outliers based on prediction errors using z-score.
    Returns boolean mask where True indicates an outlier.
    """
    errors = np.abs(y_true - y_pred)
    z_scores = (errors - errors.mean()) / errors.std()
    outliers = np.abs(z_scores) > threshold
    return outliers


def extract_absorption_data(filepath, threshold=None, normalize=True, max_peaks=None):
    """
    Extracts up to max_peaks highest absorption peaks from a .abs.txt file.
    """
    if threshold is None:
        threshold = config.ABS_THRESHOLD
    if max_peaks is None:
        max_peaks = config.MAX_PEAKS

    try:
        with open(filepath, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []

    # Find header starting with "Wavelength"
    header_index = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("Wavelength")),
        None,
    )
    if header_index is None:
        logger.warning(f"No header starting with 'Wavelength' in {filepath}.")
        return []

    wavelengths, absorptions = [], []
    for line in lines[header_index + 1 :]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                wavelengths.append(float(parts[0]))
                absorptions.append(float(parts[1]))
            except ValueError:
                logger.warning(f"Skipping line in {filepath}: {line.strip()}")
                continue

    if not wavelengths or not absorptions:
        logger.warning(f"No valid data found in {filepath}.")
        return []

    wavelengths = np.array(wavelengths)
    absorptions = np.array(absorptions)

    if normalize:
        max_val = absorptions.max()
        if max_val > 0:
            absorptions = absorptions / max_val
        else:
            logger.warning(
                f"Max absorption is zero in {filepath}; skipping normalization."
            )

    peaks, _ = find_peaks(absorptions, height=threshold)
    if peaks.size == 0:
        logger.info(f"No peaks found in {filepath} with threshold {threshold}.")
        return []

    # Build a list of tuples and sort by absorption intensity descending
    peak_data = [(wavelengths[i], absorptions[i]) for i in peaks]
    peak_data.sort(key=lambda x: x[1], reverse=True)
    return peak_data[:max_peaks]


def get_smiles_from_pubchem(
    identifier, identifier_type="name", max_retries=3, retry_delay=2
):
    """
    Retrieves a SMILES string from PubChem using a name, CAS, or CID identifier.
    Uses caching to avoid repeated API calls.
    """
    cache_key = f"{identifier_type}:{identifier}"

    with smiles_cache_lock:
        if cache_key in smiles_cache:
            return smiles_cache[cache_key]

    identifier = identifier.replace("_", " ")
    if identifier_type == "cid" and not identifier.isdigit():
        logger.error(f"Invalid CID '{identifier}'. It must be numeric.")
        return None

    url_template = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/property/IsomericSMILES/txt"
    lookup_type = "cid" if identifier_type == "cid" else "name"
    url = url_template.format(f"{lookup_type}/{identifier}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                result = response.text.strip()
                with smiles_cache_lock:
                    smiles_cache[cache_key] = result
                return result
            else:
                logger.warning(
                    f"Attempt {attempt+1}: Failed to retrieve data for {identifier} (status {response.status_code})."
                )
                if attempt == max_retries - 1:
                    with smiles_cache_lock:
                        smiles_cache[cache_key] = None
                    return None
        except (NotFoundError, PubChemHTTPError) as e:
            logger.error(f"PubChem error for {identifier}: {e}")
            if "503" in str(e):
                logger.info(
                    f"Service unavailable; retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                with smiles_cache_lock:
                    smiles_cache[cache_key] = None
                return None
        except Exception as e:
            logger.error(f"Unexpected error for {identifier}: {e}")
            with smiles_cache_lock:
                smiles_cache[cache_key] = None
            return None

    logger.error(
        f"Failed to retrieve SMILES for {identifier} after {max_retries} attempts."
    )
    with smiles_cache_lock:
        smiles_cache[cache_key] = None
    return None


def wavelength_to_ev(wavelength_nm):
    """Convert wavelength in nanometers to energy in electron volts."""
    return 1240 / wavelength_nm


def mean_absolute_error_ev(y_true_nm, y_pred_nm):
    """Calculate mean absolute error in eV."""
    y_true_ev = wavelength_to_ev(y_true_nm)
    y_pred_ev = wavelength_to_ev(y_pred_nm)
    return np.mean(np.abs(y_true_ev - y_pred_ev))


def get_smiles_batch(identifiers, identifier_type="name", max_workers=10):
    """
    Retrieve SMILES for a batch of identifiers in parallel.

    Args:
        identifiers: List of identifiers (names, CAS numbers, etc.)
        identifier_type: Type of identifier ('name', 'cas', 'cid')
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary mapping identifiers to their SMILES strings or None if not found
    """
    results = {}

    def fetch_smiles(identifier):
        return identifier, get_smiles_from_pubchem(identifier, identifier_type)

    # Parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_smiles, id) for id in identifiers]
        for future in futures:
            identifier, smiles = future.result()
            results[identifier] = smiles

    return results


def bitstring_to_array(bitstring):
    """Convert a fingerprint bitstring to a numpy array."""
    return np.array([int(bit) for bit in bitstring])
