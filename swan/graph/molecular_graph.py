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
    atom_bond_features = torch.from_numpy(
        np.hstack(generate_molecular_features(mol)).astype(np.float32))
    edges = torch.from_numpy(compute_molecular_graph_edges(mol))

    # Normalize the features
    l2_features = normalize(atom_bond_features, norm="l2", axis=1)

    return Data(x=l2_features, edge_index=edges, y=label)
