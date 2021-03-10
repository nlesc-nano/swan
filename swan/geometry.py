"""Module to help building the geometric representations."""
import json
from typing import List, Tuple

import numpy as np
from rdkit import Chem


def read_geometries_from_files(file_geometries: str) -> Tuple[List[Chem.rdchem.Mol], List[np.ndarray]]:
    """Read the molecular geometries from a file."""
    with open(file_geometries, 'r') as handler:
        strings = json.load(handler)

    molecules = [Chem.MolFromPDBBlock(s, sanitize=False) for s in strings]
    positions = [mol.GetConformer().GetPositions() for mol in molecules]
    return molecules, positions
