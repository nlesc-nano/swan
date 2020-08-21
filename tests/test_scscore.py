"""Test the SCScore class."""

import numpy as np
import pandas as pd
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

    return molecules.rdkit_molecules


def compute_scores(molecules: pd.Series, name_model: str, nbits: int) -> None:
    """Predicate score for molecules using `name_model` and `nbits`."""
    print("model name: ", name_model)
    fingerprints = generate_fingerprints(molecules, "morgan", nbits, use_chirality=True)
    print("fingerprints shape: ", fingerprints.shape)
    scorer = SCScorer(name_model)

    xs = scorer.compute_score(fingerprints)
    print("scores: ", xs)
    assert np.all(xs < 2)


def test_scscore():
    """Test that the SCScore model is properly load."""
    molecules = create_molecules_dataframe()
    compute_scores(molecules, '1024bool', 1024)
    compute_scores(molecules, '2048bool', 2048)