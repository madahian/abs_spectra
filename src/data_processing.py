#!/usr/bin/env python3
"""
Data processing pipeline for generating processed molecular data CSVs.
Uses functions from utils.py to extract data from raw files.
"""

import os
import re
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import math

import config
from src.utils import extract_absorption_data, get_smiles_batch


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.formatter = logging.Formatter("%(levelname)s: %(message)s")

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(TqdmLoggingHandler())


def detect_outliers(df, column, threshold=1.5):
    """
    Detect outliers using the IQR method.
    Returns a boolean mask where True indicates an outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    if outliers.any():
        logging.info(f"Detected {outliers.sum()} outliers in {column}")
        logging.info(f"Value range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    return outliers


def extract_molecular_data(
    raw_dir, output_csv, max_peaks=None, batch_size=50, max_workers=10
):
    """
    Process raw absorption files from raw_dir and generate a CSV with molecular data.
    Uses batch processing for SMILES retrieval and parallel processing for file handling.

    Args:
        raw_dir: Directory containing raw absorption files
        output_csv: Path to save the processed CSV file
        max_peaks: Maximum number of absorption peaks to extract per file
        batch_size: Number of molecules to process in a batch for SMILES retrieval
        max_workers: Maximum number of parallel workers for file processing
    """
    if max_peaks is None:
        max_peaks = config.MAX_PEAKS

    data = []
    processed_molecules = set()
    files = [f for f in os.listdir(raw_dir) if f.endswith(".abs.txt")]
    total_files = len(files)
    molecule_info = []
    with tqdm(
        total=total_files, desc="Parsing files", unit="file", position=0, leave=False
    ) as pbar:
        for filename in files:
            filepath = os.path.join(raw_dir, filename)
            match = re.match(r"([A-Z]\d+)_((\d+-)+\d+)_(.+?)\.abs\.txt", filename)
            if not match:
                logging.warning(f"Skipping file {filename} due to invalid format.")
                pbar.update(1)
                continue

            molecule_code = match.group(1)
            molecule_id = match.group(2)
            molecule_name = match.group(4)

            if molecule_name in processed_molecules:
                pbar.update(1)
                continue

            processed_molecules.add(molecule_name)
            molecule_info.append(
                {
                    "filepath": filepath,
                    "filename": filename,
                    "molecule_code": molecule_code,
                    "molecule_id": molecule_id,
                    "molecule_name": molecule_name,
                }
            )
            pbar.update(1)

    # Process molecules in batches to retrieve SMILES
    total_molecules = len(molecule_info)
    num_batches = math.ceil(total_molecules / batch_size)

    logging.info(
        f"Processing {total_molecules} unique molecules in {num_batches} batches"
    )

    # Processing individual molecules
    def process_molecule(info, smiles_data):
        filepath = info["filepath"]
        filename = info["filename"]
        molecule_code = info["molecule_code"]
        molecule_id = info["molecule_id"]
        molecule_name = info["molecule_name"]

        # Get SMILES from the batch results
        smiles = smiles_data.get(molecule_id)
        if smiles is None:
            smiles = smiles_data.get(molecule_name)

        if smiles is None:
            logging.warning(
                f"Could not retrieve SMILES for {molecule_name}. Skipping the molecule."
            )
            return []

        smiles_list = [s.strip() for s in smiles.splitlines() if s.strip()]
        peaks = extract_absorption_data(filepath, max_peaks=max_peaks)

        result = []
        if peaks and isinstance(peaks, list) and len(peaks) > 0:
            for smile in smiles_list:
                for wavelength, absorption in peaks:
                    result.append(
                        {
                            "Molecule Code": molecule_code,
                            "Molecule CAS": molecule_id,
                            "Molecule Name": molecule_name.replace("_", " "),
                            "SMILES": smile,
                            "Absorption Maxima": absorption,
                            "Wavelength": wavelength,
                        }
                    )
        else:
            logging.warning(
                f"No valid peaks found in {filename} or error during extraction."
            )

        return result

    with tqdm(
        total=total_molecules,
        desc="Processing molecules",
        unit="molecule",
        position=0,
        leave=True,
    ) as pbar:
        for i in range(0, total_molecules, batch_size):
            batch = molecule_info[i : i + batch_size]

            # Get SMILES for this batch using CAS IDs
            cas_ids = [m["molecule_id"] for m in batch]
            smiles_by_cas = get_smiles_batch(cas_ids, identifier_type="cas")

            # For molecules without SMILES, try names
            missing_molecules = [
                m for m in batch if smiles_by_cas.get(m["molecule_id"]) is None
            ]
            names = [m["molecule_name"] for m in missing_molecules]

            smiles_by_name = {}
            if names:
                smiles_by_name = get_smiles_batch(names, identifier_type="name")

            # Combine the results
            combined_smiles = {**smiles_by_cas, **smiles_by_name}

            # Process each molecule in parallel
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_molecule, info, combined_smiles)
                    for info in batch
                ]
                for future in futures:
                    batch_results.extend(future.result())
                    pbar.update(1)

            data.extend(batch_results)

    df = pd.DataFrame(data)

    if not df.empty:
        # Detect outliers in absorption data
        wavelength_outliers = detect_outliers(df, "Wavelength")
        absorption_outliers = detect_outliers(df, "Absorption Maxima")

        # Add outlier flags to DataFrame
        df["Wavelength_Outlier"] = wavelength_outliers
        df["Absorption_Outlier"] = absorption_outliers

        df.to_csv(output_csv, index=False)
        successful_molecules = len(set(df["Molecule Name"]))
        logging.info(
            f"{successful_molecules}/{total_files} molecules successfully processed."
        )
    else:
        logging.warning("No data was processed. The output CSV will be empty.")
        df.to_csv(output_csv, index=False)


def canonicalize_smiles(smiles):
    """
    Convert a SMILES string to canonical form using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def generate_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = mg.GetFingerprint(mol)
    return fp.ToBitString()


def process_molecular_csv(
    input_csv,
    output_train_csv,
    output_test_csv,
    test_size=0.2,
    random_state=42,
):
    """
    Process the molecular_data.csv to generate training and testing datasets.
    Includes outlier detection and handling.
    """
    df = pd.read_csv(input_csv)
    processed_data = []

    for cas, group in df.groupby("Molecule CAS"):
        unique_canon_smiles = []
        for smiles in group["SMILES"].unique():
            cs = canonicalize_smiles(smiles)
            if cs and cs not in unique_canon_smiles:
                unique_canon_smiles.append(cs)
        if not unique_canon_smiles:
            logging.warning(f"Skipping CAS {cas}: No valid SMILES found.")
            continue

        fingerprint = generate_fingerprint(unique_canon_smiles[0])
        if not fingerprint:
            logging.warning(
                f"Skipping CAS {cas}: Unable to generate fingerprint for {unique_canon_smiles[0]}."
            )
            continue

        absorption_data = group[["Absorption Maxima", "Wavelength"]].values
        # Get wavelength corresponding to maximum absorption
        primary = absorption_data[absorption_data[:, 0].argmax()][1]

        # Check if primary wavelength was marked as outlier
        is_outlier = group.loc[
            group["Wavelength"] == primary, "Wavelength_Outlier"
        ].iloc[0]

        processed_data.append(
            {
                "Molecule CAS": cas,
                "MorganFingerprint": fingerprint,
                "PrimaryWavelength": primary,
                "Is_Outlier": is_outlier,
            }
        )

    processed_df = pd.DataFrame(processed_data)

    # Split data ensuring outliers are distributed between train and test sets
    unique_cas = processed_df["Molecule CAS"].unique()
    train_cas, test_cas = train_test_split(
        unique_cas,
        test_size=test_size,
        random_state=random_state,
        stratify=processed_df.groupby("Molecule CAS")["Is_Outlier"].first(),
    )
    train_df = processed_df[processed_df["Molecule CAS"].isin(train_cas)]
    test_df = processed_df[processed_df["Molecule CAS"].isin(test_cas)]

    # Log outlier statistics
    logging.info(f"Total outliers in dataset: {processed_df['Is_Outlier'].sum()}")
    logging.info(f"Outliers in training set: {train_df['Is_Outlier'].sum()}")
    logging.info(f"Outliers in test set: {test_df['Is_Outlier'].sum()}")

    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)


def main():
    raw_dir = os.path.join(config.RAW_DATA_DIR, "PhotochemCAD", "Common Compounds")
    processed_dir = config.PROCESSED_DATA_DIR
    os.makedirs(processed_dir, exist_ok=True)

    molecular_data_csv = config.MOLECULAR_DATA_CSV
    training_data_csv = config.TRAIN_DATA_CSV
    test_data_csv = config.TEST_DATA_CSV

    extract_molecular_data(
        raw_dir,
        molecular_data_csv,
        max_peaks=config.MAX_PEAKS,
        batch_size=50,
        max_workers=8,
    )
    process_molecular_csv(molecular_data_csv, training_data_csv, test_data_csv)


if __name__ == "__main__":
    main()
