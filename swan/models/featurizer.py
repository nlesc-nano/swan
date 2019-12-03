"""Compute the fingerprints of an array of smiles."""

from functools import partial
from itertools import chain
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D


def generate_fingerprints(molecules: pd.Series, radius: int = 2, bits: int = 2000) -> np.ndarray:
    """Generate the Extended-Connectivity Fingerprints (ECFP).

    Use the method described at: https://doi.org/10.1021/ci100050t
    """
    size = len(molecules)
    chunks = size // cpu_count() if size > cpu_count() else 1
    worker = partial(compute_fingerprint, molecules, radius, bits)
    with Pool() as p:
        it = p.imap(worker, molecules.index, chunks)
        fingerprints = np.fromiter(chain.from_iterable(it), np.float32)

    # fingerprints = [compute_fingerprint(mol, radius, bits) for mol in molecules]
    return fingerprints.reshape(size, bits)


def compute_fingerprint(molecules, radius: int, nBits: int, index: int) -> np.ndarray:
    """Calculate a single fingerprint."""
    fp = AllChem.GetMorganFingerprintAsBitVect(molecules[index], radius, nBits)
    return np.fromiter((float(k) for k in fp.ToBitString()), np.float32, nBits)


def create_molecules(smiles: np.array) -> list:
    """Create a list of RDKit molecules."""
    return [Chem.MolFromSmiles(s) for s in smiles]


def compute_3D_descriptors(molecules: list) -> np.array:
    """Compute the Asphericity and Eccentricity for an array of molecules."""
    asphericity = np.fromiter((Descriptors3D.Asphericity(m) for m in molecules), np.float32)
    eccentricity = np.fromiter((Descriptors3D.Eccentricity(m) for m in molecules), np.float32)

    return np.stack((asphericity, eccentricity)).T
