#!/usr/bin/env python3

"""
    This script reads the 'molecular_data.csv' file from the 'data' folder,
    converts each molecule's SMILES into a Morgan fingerprint, and retains
    only [Molecule CAS, MorganFingerprint, Wavelength, Absorption Maxima].
    To avoid data leakage, it splits the dataset based on unique molecules
    (identified by Molecule CAS). All rows for a given molecule end up
    exclusively in either training or test set.

Usage:
    python generate_morgan_fingerprints.py
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    root_dir = get_project_root()

    # 1. Define input and output paths relative to the root
    input_csv = os.path.join(root_dir, "data", "molecular_data.csv")
    train_csv = os.path.join(root_dir, "data", "training_data.csv")
    test_csv = os.path.join(root_dir, "data", "test_data.csv")

    # 2. Load the dataset
    df = pd.read_csv(input_csv)

    # 3. Generate Morgan fingerprints and store in a new column
    fingerprints = []
    for idx, row in df.iterrows():
        smiles = row["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Setup: radius=2, 1024-bit length
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            bitstring = fp.ToBitString()
            fingerprints.append(bitstring)
        else:
            fingerprints.append(None)

    df["MorganFingerprint"] = fingerprints

    # 4. Keep only Molecule CAS, MorganFingerprint, Wavelength, and Absorption Maxima
    df = df[["Molecule CAS", "MorganFingerprint", "Wavelength", "Absorption Maxima"]]

    # 5. Split at the *molecule level* to avoid leakage
    #    - First, get unique CAS numbers.
    unique_cas = df["Molecule CAS"].unique()

    #    - Split unique CAS into train and test sets.
    train_cas, test_cas = train_test_split(unique_cas, test_size=0.2, random_state=42)

    #    - Filter the original DataFrame so all rows for a given CAS are in train OR test.
    train_df = df[df["Molecule CAS"].isin(train_cas)]
    test_df = df[df["Molecule CAS"].isin(test_cas)]

    # 6. Save the split datasets
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Training data saved to {train_csv}")
    print(f"Test data saved to {test_csv}")


if __name__ == "__main__":
    main()
