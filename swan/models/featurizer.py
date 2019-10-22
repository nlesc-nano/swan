from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
import numpy as np


def generate_fingerprints(smiles, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """
    Generate the Extended-Connectivity Fingerprints (ECFP) for the `smiles`
    using the method described at: https://doi.org/10.1021/ci100050t
    """
    chunks = smiles.size // cpu_count()
    worker = partial(compute_fingerprint, smiles, radius, nBits)
    with Pool() as p:
        result = list(p.imap(worker, range(smiles.size), chunks))

    return np.stack(result)


def compute_fingerprint(smiles, radius: int, nBits: int, index: int) -> np.ndarray:
    """
    Calculate a single fingerprint
    """
    mol = Chem.MolFromSmiles(smiles[index])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array([int(k) for k in fp.ToBitString()])
