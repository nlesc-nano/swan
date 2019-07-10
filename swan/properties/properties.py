from rdkit import Chem
import numpy as np


def compute_mol_shape(smile: str, boxDim: tuple = (20, 20, 20), spacing=0.5) -> np.ndarray:
    """
    Compute a grid representing the molecular shape
    """
    # Create a canonical molecule
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol)
    Chem.rdMolTransforms.CanonicalizeMol(mol)

    # compute grid
    grid = Chem.AllChem.ComputeMolShape(mol, boxDim=boxDim, spacing=spacing)

    # Create an array to contain the grid
    size = grid.GetSize()
    arr = np.empty(size)

    for i in range(size):
        arr[i] = grid.GetVal(i)

    return arr
