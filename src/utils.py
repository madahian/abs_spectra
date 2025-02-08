import time
import logging
import requests
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pubchempy import get_compounds, NotFoundError, PubChemHTTPError

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
        logging.error(f"File not found: {filepath}")
        return []

    # Find header starting with "Wavelength"
    header_index = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("Wavelength")),
        None,
    )
    if header_index is None:
        logging.warning(f"No header starting with 'Wavelength' in {filepath}.")
        return []

    wavelengths, absorptions = [], []
    for line in lines[header_index + 1 :]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                wavelengths.append(float(parts[0]))
                absorptions.append(float(parts[1]))
            except ValueError:
                logging.warning(f"Skipping line in {filepath}: {line.strip()}")
                continue

    if not wavelengths or not absorptions:
        logging.warning(f"No valid data found in {filepath}.")
        return []

    wavelengths = np.array(wavelengths)
    absorptions = np.array(absorptions)

    if normalize:
        max_val = absorptions.max()
        if max_val > 0:
            absorptions = absorptions / max_val
        else:
            logging.warning(
                f"Max absorption is zero in {filepath}; skipping normalization."
            )

    peaks, _ = find_peaks(absorptions, height=threshold)
    if peaks.size == 0:
        logging.info(f"No peaks found in {filepath} with threshold {threshold}.")
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
    """
    identifier = identifier.replace("_", " ")
    if identifier_type == "cid" and not identifier.isdigit():
        logging.error(f"Invalid CID '{identifier}'. It must be numeric.")
        return None

    url_template = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/property/IsomericSMILES/txt"
    lookup_type = "cid" if identifier_type == "cid" else "name"
    url = url_template.format(f"{lookup_type}/{identifier}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.text.strip()
            else:
                logging.warning(
                    f"Attempt {attempt+1}: Failed to retrieve data for {identifier} (status {response.status_code})."
                )
                return None
        except (NotFoundError, PubChemHTTPError) as e:
            logging.error(f"PubChem error for {identifier}: {e}")
            if "503" in str(e):
                logging.info(
                    f"Service unavailable; retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                return None
        except Exception as e:
            logging.error(f"Unexpected error for {identifier}: {e}")
            return None

    logging.error(
        f"Failed to retrieve SMILES for {identifier} after {max_retries} attempts."
    )
    return None
