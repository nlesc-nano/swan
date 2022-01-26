"""Module to help building the geometric representations."""
import json
import multiprocessing
from functools import partial
from typing import Collection, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from swan.type_hints import PathLike


def read_geometries_from_files(file_geometries: PathLike) -> Tuple[List[Chem.rdchem.Mol], List[np.ndarray]]:
    """Read the molecular geometries from a file.

    Parameters
    ----------
    file_geometries
        Path to the files with the geometries in JSON format

    Returns
    -------
    Tuple with a list of rdkit geometries and a list of matrices with the molecular geometries.

    """
    with open(file_geometries, 'r') as handler:
        strings = json.load(handler)

    molecules = [Chem.MolFromPDBBlock(s, sanitize=False) for s in strings]
    positions = [np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32) for mol in molecules]
    return molecules, positions


def guess_positions(molecules: Collection[Chem.rdchem.Mol], optimize_molecule: bool) -> List[np.ndarray]:
    """Compute a guess for the molecular coordinates.

    Parameters
    ----------
    molecules
        Collection containing the molecules
    optimize_molecule
        Whether or not to perform a molecular optimization

    Returns
    -------
    List of the molecular coordinates

    """
    chunksize = 2 * len(molecules) // multiprocessing.cpu_count()
    function = partial(get_coordinates, optimize_molecule)
    with multiprocessing.Pool() as p:
        data = list(p.imap(function, molecules, chunksize=chunksize))

    return data


def get_coordinates(optimize_molecule: bool, mol: Chem.rdchem.Mol) -> np.ndarray:
    """Extract the coordinates of a given RDKit molecule.
    Parameters
    ----------
    optimize_molecule
        Whether or not to perform a molecular optimization
    mol
        RDKit molecule

    Returns
    -------
    Array with the molecular coordinates

    """
    # AllChem.EmbedMolecule(mol)
    if optimize_molecule:
        AllChem.UFFOptimizeMolecule(mol)
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
