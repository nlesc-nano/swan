"""Test the features generation functionality."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from swan.dataset.features.featurizer import (compute_molecular_graph_edges,
                                              generate_molecular_features)

MOL = Chem.MolFromSmiles("CC(=O)O")
AllChem.EmbedMolecule(MOL)


def test_molecular_features():
    """Test that the atomic and bond features are properly created."""
    # Generate mol and add conformers

    atomic, bond = generate_molecular_features(MOL)

    # There are four heavy atoms with 18 atomic features each
    assert atomic.shape == (4, 18)

    # There are 3 x 2 bidirectional edges (Bonds) with 7 features each
    assert bond.shape == (6, 7)

    # All entries in the matrix are different of Nan
    assert not np.all(np.isnan(atomic))
    assert not np.all(np.isnan(bond))


def test_molecular_graph():
    """Test that the molecular graph is correctly generated."""
    graph = compute_molecular_graph_edges(MOL)
    expected = np.array([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]], dtype=int)

    assert np.all(graph == expected)
