"""Module to compute the atomic features."""

import mendeleev
import numpy as np
from rdkit.Chem import rdchem

__all__ = ["dict_element_features"]

ELEMENTS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = [rdchem.BondType.SINGLE, rdchem.BondType.AROMATIC,
         rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE]


def generate_atomic_features(symbol: str) -> np.array:
    """Get the features for a single atom."""
    len_elements = len(ELEMENTS)
    features = np.zeros(len_elements + 4)
    el = mendeleev.element(symbol)
    atom_type_index = ELEMENTS.index(symbol)
    features[atom_type_index] = 1  # Bondtype
    features[len_elements] = el.vdw_radius  # Van der Waals radius
    features[len_elements + 1] = el.covalent_radius
    features[len_elements + 2] = el.electronegativity()

    return features


dict_element_features = {el: generate_atomic_features(el) for el in ELEMENTS}
