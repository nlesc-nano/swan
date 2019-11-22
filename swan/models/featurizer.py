"""Compute the fingerprints of an array of smiles."""
from functools import partial
from multiprocessing import (Pool, cpu_count)
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def generate_fingerprints(smiles, radius: int = 2, bits: int = 2048) -> np.ndarray:
    """
    Generate the Extended-Connectivity Fingerprints (ECFP) for the `smiles`
    using the method described at: https://doi.org/10.1021/ci100050t
    """
    chunks = smiles.size // cpu_count() if smiles.size > cpu_count() else 1
    worker = partial(compute_fingerprint, smiles, radius, bits)
    with Pool() as p:
        result = list(p.imap(worker, range(smiles.size), chunks))

    return np.stack(result)


def compute_fingerprint(smiles, radius: int, nBits: int, index: int) -> np.ndarray:
    """Calculate a single fingerprint."""
    mol = Chem.MolFromSmiles(smiles[index])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array([int(k) for k in fp.ToBitString()], dtype=np.float32)
