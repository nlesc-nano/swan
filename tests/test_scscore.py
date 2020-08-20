"""Test the SCScore class."""

import numpy as np

from rdkit import Chem
from swan.features.featurizer import generate_fingerprints
from swan.models.scscore import SCScorer
from swan.utils import read_molecules
from .utils_test import PATH_TEST

PATH_SMILES = PATH_TEST / "smiles_carboxylic.csv"


def create_molecules_dataframe():
    """Read the smiles file and generate a pandas DataFrame from it."""
    molecules = read_molecules(PATH_SMILES)
    converter = np.vectorize(Chem.MolFromSmiles)
    molecules["rdkit_molecules"] = converter(molecules.smiles)

    return molecules


def test_scscore():
    """Test that the SCScore model is properly load."""
    molecules = create_molecules_dataframe()
    fingerprints = generate_fingerprints(molecules.rdkit_molecules, "morgan", 1024, use_chirality=True)
    scorer = SCScorer('1024bool')
    for k in range(len(molecules)):
        x = scorer.compute_score(fingerprints[k])
        print(f"score: {x} {molecules.smiles[k]}")
