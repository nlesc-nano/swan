"""Module to help building the geometric representations."""
import json
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Union
from functools import partial

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

PathLike = Union[str, Path]
DTYPE = np.float32


def read_geometries_from_files(file_geometries: PathLike) -> Tuple[List[Chem.rdchem.Mol], List[np.ndarray]]:
    """Read the molecular geometries from a file."""
    with open(file_geometries, 'r') as handler:
        strings = json.load(handler)

    molecules = [Chem.MolFromPDBBlock(s, sanitize=False) for s in strings]
    positions = [np.asarray(mol.GetConformer().GetPositions(), dtype=DTYPE) for mol in molecules]
    return molecules, positions


def guess_positions(molecules: pd.Series, optimize_molecule: bool) -> List[np.ndarray]:
    """Compute a guess for the molecular coordinates."""
    chunksize = 2 * len(molecules) // multiprocessing.cpu_count()
    function = partial(get_coordinates, optimize_molecule)
    with multiprocessing.Pool() as p:
        data = list(p.imap(function, molecules, chunksize=chunksize))

    return data


def get_coordinates(optimize_molecule: bool, mol: Chem.rdchem.Mol) -> np.ndarray:
    # AllChem.EmbedMolecule(mol)
    if optimize_molecule:
        AllChem.UFFOptimizeMolecule(mol)
    return mol.GetConformer().GetPositions()
