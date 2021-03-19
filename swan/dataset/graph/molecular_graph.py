"""Generation of molecular graphs.

Index
-----
.. currentmodule:: swan.graph.molecular_graph
.. autosummary::
    create_molecular_torch_geometric_graph


API
---
.. autofunction:: create_molecular_torch_geometric_graph

"""
import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
import torch_geometric as tg

from swan.dataset.features.featurizer import (compute_molecular_graph_edges,
                                              generate_molecular_features)


def create_molecular_torch_geometric_graph(
        mol: Chem.rdchem.Mol, positions: Tensor = None, labels: Tensor = None) -> tg.data.Data:
    """Create a torch-geometry data object representing a graph.

    See torch-geometry documentation:
    https://pytorch-geometric.readthedocs.io/en/latest/?badge=latest
    The graph nodes contains atomic and bond pair information.
    """
    atomic_features, bond_features = generate_molecular_features(mol)
    # Undirectional edges to represent molecular bonds
    edges = torch.from_numpy(compute_molecular_graph_edges(mol))

    atomic_features = torch.tensor(atomic_features.astype(np.float32))
    bond_features = torch.tensor(bond_features.astype(np.float32))

    return tg.data.Data(
        x=atomic_features,
        edge_attr=bond_features,
        edge_index=edges,
        positions=positions,
        y=labels)
