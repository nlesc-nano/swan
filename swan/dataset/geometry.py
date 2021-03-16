"""Module to help building the geometric representations."""
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from rdkit import Chem

PathLike = Union[str, Path]


def read_geometries_from_files(file_geometries: PathLike) -> Tuple[List[Chem.rdchem.Mol], List[np.ndarray]]:
    """Read the molecular geometries from a file."""
    with open(file_geometries, 'r') as handler:
        strings = json.load(handler)

    molecules = [Chem.MolFromPDBBlock(s, sanitize=False) for s in strings]
    positions = [mol.GetConformer().GetPositions() for mol in molecules]
    return molecules, positions
