"""Generation of molecular graphs."""
import numpy as np
import torch
from rdkit import Chem
# import torch_geometric as tg
from sklearn.preprocessing import normalize
from torch import Tensor
from torch_geometric.data import Data

from ..features.featurizer import (compute_molecular_graph_edges,
                                   generate_molecular_features)


def create_molecular_graph_data(mol: Chem.rdchem.Mol, label: Tensor) -> Data:
    """Create a torch-geometry data object representing a graph.

    See torch-geometry documentation:
    https://pytorch-geometric.readthedocs.io/en/latest/?badge=latest
    The graph nodes contains atomic and bond pair information.
    """
    atomic_features, bond_features = generate_molecular_features(mol)
    edges = torch.from_numpy(compute_molecular_graph_edges(mol))

    # Normalize the features
    l2_atomic_features = torch.from_numpy(
        normalize(atomic_features, norm="l2", axis=1).astype(np.float32))
    l2_bond_features = torch.from_numpy(
        normalize(bond_features, norm="l2", axis=1).astype(np.float32))

    return Data(x=l2_atomic_features, edge_attr=l2_bond_features, edge_index=edges, y=label)
