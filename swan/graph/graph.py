"""Module to generate Molecular graphs."""

import numpy as np
from rdkit import Chem

atomic_symbols = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
    'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn']

index_symbols = {x: (i + 1) for i, x in enumerate(atomic_symbols)}


def create_connectivity_graph(smile: str) -> np.array:
    """Create a 3D tensor representing a connectivity graph.

    The tensor size is 2 x N x N, where N is the number of atoms.
    """
    mol = Chem.MolFromSmiles(smile)
    return Chem.rdmolops.GetAdjacencyMatrix(mol)