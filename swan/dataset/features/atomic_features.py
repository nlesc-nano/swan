"""Module to compute the atomic features."""

import mendeleev
import numpy as np
from rdkit.Chem import rdchem

__all__ = ["BONDS", "ELEMENTS", "dict_element_features", "compute_hybridization_index"]

ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = [rdchem.BondType.SINGLE, rdchem.BondType.AROMATIC,
         rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE]

hybridization = {rdchem.HybridizationType.SP: 0,
                 rdchem.HybridizationType.SP2: 1,
                 rdchem.HybridizationType.SP3: 2}


def generate_atomic_features(symbol: str) -> np.ndarray:
    """Get the features for a single atom.

    parameters
    ----------
    symbol
        Atomic symbol of a given atom

    Returns
    -------
    Numpy array with the atomic features for the given atom type

    """
    len_elements = len(ELEMENTS)
    features = np.zeros(len_elements + 3)
    el = mendeleev.element(symbol)
    atom_type_index = ELEMENTS.index(symbol)
    features[atom_type_index] = 1  # Bondtype
    features[len_elements] = el.vdw_radius  # Van der Waals radius
    features[len_elements + 1] = el.covalent_radius
    features[len_elements + 2] = el.electronegativity()

    return features


def compute_hybridization_index(atom: rdchem.Atom) -> float:
    """Return whether the atoms' hybridization is: SP, SP2, SP3 or Other.

    Parameters
    ----------
    atom
        RDKit atom representation

    Returns
    -------
    float where: 1 => SP; 2 => SP2; 3 => SP3; other => 4

    """
    hyb = atom.GetHybridization()
    index = hybridization.get(hyb)
    return index if index is not None else 4


dict_element_features = {el: generate_atomic_features(el) for el in ELEMENTS}
