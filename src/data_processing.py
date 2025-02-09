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

import config
from src.utils import extract_absorption_data, get_smiles_from_pubchem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_molecular_data(raw_dir, output_csv, max_peaks=None):
    """
    Process raw absorption files from raw_dir and generate a CSV with molecular data.
    """
    if max_peaks is None:
        max_peaks = config.MAX_PEAKS
    data = []
    processed_molecules = set()
    files = [f for f in os.listdir(raw_dir) if f.endswith(".abs.txt")]

    for filename in tqdm(files, desc="Extracting molecular data", unit="file"):
        if filename.endswith(".abs.txt"):
            filepath = os.path.join(raw_dir, filename)
            match = re.match(r"([A-Z]\d+)_((\d+-)+\d+)_(.+?)\.abs\.txt", filename)
            if not match:
                logging.warning(f"Skipping file {filename} due to invalid format.")
                continue

            molecule_code = match.group(1)
            molecule_id = match.group(2)
            molecule_name = match.group(4)

            if molecule_name in processed_molecules:
                continue
            processed_molecules.add(molecule_name)

            smiles = get_smiles_from_pubchem(molecule_id, identifier_type="cas")
            if smiles is None:
                smiles = get_smiles_from_pubchem(molecule_name, identifier_type="name")
            if smiles is None:
                logging.warning(
                    f"Could not retrieve SMILES for {molecule_name}. Skipping the molecule."
                )
                continue

            smiles_list = [s.strip() for s in smiles.splitlines() if s.strip()]
            peaks = extract_absorption_data(filepath, max_peaks=max_peaks)
            if peaks and isinstance(peaks, list) and len(peaks) > 0:
                for smile in smiles_list:
                    for wavelength, absorption in peaks:
                        data.append(
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

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    successful_molecules = len(set(df['Molecule Name']))
    total_molecules = len(files)
    logging.info(f"{successful_molecules}/{total_molecules} molecules successfully processed.")


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
    min_distance_nm=50,
    test_size=0.2,
    random_state=42,
):
    """
    Process the molecular_data.csv to generate training and testing datasets.
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
        sorted_data = sorted(absorption_data, key=lambda x: x[0], reverse=True)

        filtered_peaks = []
        for abs_val, wl in sorted_data:
            if all(abs(wl - sel_wl) >= min_distance_nm for sel_wl in filtered_peaks):
                filtered_peaks.append(wl)
            if len(filtered_peaks) >= 3:
                break

        primary = filtered_peaks[0] if len(filtered_peaks) >= 1 else None
        secondary1 = filtered_peaks[1] if len(filtered_peaks) >= 2 else None
        secondary2 = filtered_peaks[2] if len(filtered_peaks) >= 3 else None

        processed_data.append(
            {
                "Molecule CAS": cas,
                "CanonicalSMILES": unique_canon_smiles[0],
                "MorganFingerprint": fingerprint,
                "PrimaryWavelength": primary,
                "SecondaryWavelength1": secondary1,
                "SecondaryWavelength2": secondary2,
            }
        )

    processed_df = pd.DataFrame(processed_data)
    unique_cas = processed_df["Molecule CAS"].unique()
    train_cas, test_cas = train_test_split(
        unique_cas, test_size=test_size, random_state=random_state
    )
    train_df = processed_df[processed_df["Molecule CAS"].isin(train_cas)]
    test_df = processed_df[processed_df["Molecule CAS"].isin(test_cas)]

    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)


def main():
    raw_dir = os.path.join(config.RAW_DATA_DIR, "PhotochemCAD", "Common Compounds")
    processed_dir = config.PROCESSED_DATA_DIR
    os.makedirs(processed_dir, exist_ok=True)

    molecular_data_csv = config.MOLECULAR_DATA_CSV
    training_data_csv = config.TRAIN_DATA_CSV
    test_data_csv = config.TEST_DATA_CSV

    extract_molecular_data(raw_dir, molecular_data_csv, max_peaks=config.MAX_PEAKS)
    process_molecular_csv(molecular_data_csv, training_data_csv, test_data_csv)


if __name__ == "__main__":
    main()
