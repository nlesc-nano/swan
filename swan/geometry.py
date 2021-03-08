"""Module to help building the geometric representations."""
from rdkit import Chem
import json
from typing import List


def read_geometries_from_files(file_geometries: str) -> List[Chem.rdchem.Mol]:
    """Read the molecular geometries from a file."""
    with open(file_geometries, 'r') as handler:
        strings = json.load(handler)

    return [Chem.MolFromPDBBlock(s, sanitize=False) for s in strings]
