#!/usr/bin/env python3

"""
    Processes molecular data to create a machine learning-ready dataset where each row represents a unique molecule.
    Handles multiple SMILES representations, extracts primary/secondary absorption wavelengths, and ensures no data leakage.

Key Features:
    1. Canonical SMILES Handling: Converts all SMILES representations of a molecule to a standardized form.
    2. Absorption Peak Extraction: Identifies the top 3 absorption wavelengths per molecule. 
        - Missing secondary peaks are stored as `NaN`.
    3. Morgan Fingerprints: Generates 1024-bit Morgan fingerprints for structural representation.
    4. Leakage-Free Splitting: Splits data at the molecule level (by CAS number) for training/testing.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split


def get_project_root():
    """Return absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def canonicalize_smiles(smiles):
    """Convert SMILES string to canonical form. Returns None for invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def main():
    root_dir = get_project_root()

    # File paths
    input_csv = os.path.join(root_dir, "data", "molecular_data.csv")
    train_csv = os.path.join(root_dir, "data", "training_data.csv")
    test_csv = os.path.join(root_dir, "data", "test_data.csv")

    # Load raw data
    df = pd.read_csv(input_csv)

    # Process each molecule group
    processed_data = []
    for cas, group in df.groupby("Molecule CAS"):
        # Step 1: Standardize SMILES representations
        unique_canon_smiles = []
        for smiles in group["SMILES"].unique():
            cs = canonicalize_smiles(smiles)
            if cs and (cs not in unique_canon_smiles):
                unique_canon_smiles.append(cs)

        if not unique_canon_smiles:
            print(f"Skipping CAS {cas}: No valid SMILES found.")
            continue

        # Step 2: Generate fingerprint from first valid canonical SMILES
        mol = Chem.MolFromSmiles(unique_canon_smiles[0])
        if not mol:
            print(
                f"Skipping CAS {cas}: Invalid canonical SMILES {unique_canon_smiles[0]}."
            )
            continue
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=1024
        ).ToBitString()

        # Step 3: Extract absorption peaks (sorted by intensity)
        absorption_data = group[["Absorption Maxima", "Wavelength"]].values
        sorted_peaks = sorted(
            absorption_data, key=lambda x: x[0], reverse=True
        )  # Sort by absorption maxima

        # Assign primary/secondary wavelengths (allow NaN for missing peaks)
        primary = sorted_peaks[0][1] if len(sorted_peaks) >= 1 else None
        secondary1 = sorted_peaks[1][1] if len(sorted_peaks) >= 2 else None
        secondary2 = sorted_peaks[2][1] if len(sorted_peaks) >= 3 else None

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

    # Create final DataFrame
    processed_df = pd.DataFrame(processed_data)

    # Split data by molecule (CAS number)
    unique_cas = processed_df["Molecule CAS"].unique()
    train_cas, test_cas = train_test_split(unique_cas, test_size=0.2, random_state=42)

    train_df = processed_df[processed_df["Molecule CAS"].isin(train_cas)]
    test_df = processed_df[processed_df["Molecule CAS"].isin(test_cas)]

    # Save datasets
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Training data ({len(train_df)} molecules) saved to {train_csv}")
    print(f"Test data ({len(test_df)} molecules) saved to {test_csv}")


if __name__ == "__main__":
    main()
