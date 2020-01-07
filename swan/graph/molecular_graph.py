"""Generation of molecular graphs."""
import numpy as np
import torch
from rdkit import Chem
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
    # Undirectional edges to represent molecular bonds
    edges = torch.from_numpy(compute_molecular_graph_edges(mol))

    atomic_features = torch.from_numpy(atomic_features.astype(np.float32))
    bond_features = torch.from_numpy(bond_features.astype(np.float32))

    return Data(x=atomic_features, edge_attr=bond_features, edge_index=edges, y=label)
