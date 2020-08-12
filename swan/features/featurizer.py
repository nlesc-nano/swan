"""Compute the fingerprints of an array of smiles."""

from itertools import chain

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from typing import Any, Tuple
from typing_extensions import Protocol

from .atomic_features import (ELEMENTS, BONDS, compute_hybridization_index, dict_element_features)


class FingerPrintCalculator(Protocol):
    """Type representing the function to compute the fingerprint."""
    def __call__(self, mol: Chem.rdchem.Mol, radious: int, nbits: int = 1024, **kwargs: Any) -> ExplicitBitVect:
        ...


dictionary_functions = {
    "morgan": AllChem.GetMorganFingerprintAsBitVect,
    "atompair": AllChem.GetHashedAtomPairFingerprintAsBitVect,
    "torsion": AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect
}

# atom_type(len_elements) + vdw + covalent_radius + electronegativity + hybridization +
# is_aromatic
NUMBER_ATOMIC_GRAPH_FEATURES = len(ELEMENTS) + 8
# Bond_type(4) + same_ring + distance
NUMBER_BOND_GRAPH_FEATURES = len(BONDS) + 3
# Concatenation of both features set
NUMBER_GRAPH_FEATURES = NUMBER_ATOMIC_GRAPH_FEATURES + NUMBER_BOND_GRAPH_FEATURES


def generate_molecular_features(mol: Chem.rdchem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """Generate both atomic and atom-pair features excluding the hydrogens.

    Atom types: C N O F P S Cl Br I.

    Atomic features,

    * Atom type: One hot vector (size 9).
    * Radius: Van der Waals and Convalent radious (size 2)
    * Electronegativity (size 1)
    * Hybridization: SP, SP2, SP3 (size 3)
    * Number of hydrogen (size 1)
    * Is Aromatic: Whether the atoms is part of an aromatic ring (size 1)

    Bond features,

    * Bond type: One hot vector of {Single,  Aromatic, Double, Triple} (size 4)
    * Same Ring: Whether the atoms are in the same ring (size 1)
    * Distance: Euclidean distance between the pair (size 1)
    """
    number_atoms = mol.GetNumAtoms()
    atomic_features = np.zeros((number_atoms, NUMBER_ATOMIC_GRAPH_FEATURES))
    len_elements = len(ELEMENTS)
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_features[i, : len_elements + 3] = dict_element_features[atom.GetSymbol()]
        hybrid_index = compute_hybridization_index(atom)
        atomic_features[i, len_elements + 3 + hybrid_index] = 1.0
        atomic_features[i, len_elements + 6] = float(atom.GetTotalNumHs())
        atomic_features[i, -1] = float(atom.GetIsAromatic())

    # Represent an undirectional graph using two arrows for each bond
    bond_features = np.zeros((2 * mol.GetNumBonds(), NUMBER_BOND_GRAPH_FEATURES))
    for i, bond in enumerate(mol.GetBonds()):
        feats = generate_bond_features(mol, bond)
        bond_features[2 * i] = feats
        bond_features[2 * i + 1] = feats

    return atomic_features.astype(np.float32), bond_features.astype(np.float32)


def generate_bond_features(mol: Chem.rdchem.Mol, bond: Chem.rdchem.Bond) -> np.ndarray:
    """Compute the features for a given bond.

    * Bond type: One hot vector of {Single,  Aromatic, Double, Triple} (size 4)
    * Same Ring: Whether the atoms are in the same ring (size 1)
    * Conjugated: Whether the bond is considered conjugated (size 1)
    * Distance: Euclidean distance between the pair (size 1)
    """
    bond_features = np.zeros(NUMBER_BOND_GRAPH_FEATURES)
    bond_type = BONDS.index(bond.GetBondType())
    bond_features[bond_type] = 1.0

    # Is the bond in the same ring
    bond_features[4] = float(bond.IsInRing())

    # Is the bond conjugated
    bond_features[5] = float(bond.GetIsConjugated())

    # Distance
    begin = bond.GetBeginAtom().GetIdx()
    end = bond.GetEndAtom().GetIdx()
    bond_features[6] = Chem.rdMolTransforms.GetBondLength(mol.GetConformer(), begin, end)

    return bond_features


def compute_molecular_graph_edges(mol: Chem.rdchem.Mol) -> np.ndarray:
    """Generate the edges for a molecule represented as a graph.

    The edges are represented as a matrix of dimension 2 X ( 2 * number_of_bonds).
    With a two edges for each bond representing a undirectional graph.
    """
    number_edges = 2 * mol.GetNumBonds()
    edges = np.zeros((2, number_edges), dtype=np.int)
    for k, bond in enumerate(mol.GetBonds()):
        edges[0, 2 * k] = bond.GetBeginAtomIdx()
        edges[1, 2 * k] = bond.GetEndAtomIdx()
        edges[0, 2 * k + 1] = bond.GetEndAtomIdx()
        edges[1, 2 * k + 1] = bond.GetBeginAtomIdx()

    return edges


def generate_fingerprints(molecules: pd.Series, fingerprint: str, bits: int) -> np.ndarray:
    """Generate the Extended-Connectivity Fingerprints (ECFP).

    Available fingerprints:
    * morgan https://doi.org/10.1021/ci100050t
    * atompair
    * torsion
    """
    size = len(molecules)
    # Select the fingerprint calculator
    fingerprint_calculator = dictionary_functions[fingerprint]

    it = (compute_fingerprint(molecules[i], fingerprint_calculator, bits) for i in molecules.index)
    result = np.fromiter(
        chain.from_iterable(it),
        np.float32,
        size * bits
    )

    return result.reshape(size, bits)


def compute_fingerprint(molecule, function: FingerPrintCalculator, nbits: int) -> np.ndarray:
    """Calculate a single fingerprint."""
    bit_vector = function(molecule, nbits)
    return np.fromiter((float(k) for k in bit_vector.ToBitString()), np.float32, nbits)
