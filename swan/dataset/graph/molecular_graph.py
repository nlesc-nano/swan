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
import dgl
import torch
import torch_geometric as tg
from rdkit import Chem
from torch import Tensor

from swan.dataset.features.featurizer import (compute_molecular_graph_edges,
                                              generate_molecular_features)


def create_molecular_torch_geometric_graph(
        mol: Chem.rdchem.Mol, positions: Tensor = None, labels: Tensor = None) -> tg.data.Data:
    """Create a torch-geometry data object representing a graph.

    See torch-geometry documentation:
    https://pytorch-geometric.readthedocs.io/en/latest/?badge=latest
    The graph nodes contains atomic and bond pair information.
    """
    atomic_features, bond_features = [
        torch.from_numpy(array) for array in generate_molecular_features(mol)]
    # Undirectional edges to represent molecular bonds
    edges = torch.from_numpy(compute_molecular_graph_edges(mol))

    return tg.data.Data(
        x=atomic_features,        # [num_atoms, NUMBER_ATOMIC_GRAPH_FEATURES]
        edge_attr=bond_features,  # [num_atoms, NUMBER_BOND_GRAPH_FEATURES]
        edge_index=edges,         # [2, 2 x num_bonds]
        positions=positions,      # [num_atoms, 3]
        y=labels)


def create_molecular_dgl_graph(
        mol: Chem.rdchem.Mol, positions: Tensor = None, labels: Tensor = None) -> dgl.DGLGraph:
    """Create a DGL Graph object.

    See: https://www.dgl.ai/
    The graph nodes contains atomic and bond pair information.

    """
    atomic_features, bond_features = [
        torch.from_numpy(array) for array in generate_molecular_features(mol)]

    # Undirectional edges to represent molecular bonds
    src, dst = torch.from_numpy(compute_molecular_graph_edges(mol))

    # Create graph
    graph = dgl.DGLGraph((src, dst))

    # Add node features to graph
    graph.ndata['x'] = positions  # [num_atoms, 3]
    graph.ndata['f'] = atomic_features  # [num_atoms, NUMBER_ATOMIC_GRAPH_FEATURES]

    # Add edge features to graph
    graph.edata['d'] = positions[dst] - positions[src]  # [num_atoms, 3]
    graph.edata['w'] = bond_features  # [num_atoms, NUMBER_BOND_GRAPH_FEATURES]

    return graph
