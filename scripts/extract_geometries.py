#!/usr/bin/env python

import argparse
import json
from typing import Any, Dict, List, Tuple

import h5py
import pandas as pd
from dataCAT import PDBContainer
from rdkit import Chem

XYZ = List[Tuple[str, float, float, float]]


def extract_data(files: List[str], csv_file: str):
    """Extract geometries from the HDF5"""
    df = pd.read_csv(csv_file, index_col=0)
    data = {}
    for f in files:
        with h5py.File(f, "r") as handler:
            data.update(get_geometries(handler))

    # Search for the corresponding geometries
    smiles = df.smiles.to_list()
    geometries = [data[s] for s in smiles]
    store_as_json_array(geometries)


def get_geometries(handler: h5py.File) -> Dict[str, List[Any]]:
    """Return an array of smiles in the HDF5."""
    group = handler["ligand"]
    smiles = group["index"]["ligand"][()]
    pdb = PDBContainer.from_hdf5(group)
    molecules = pdb.to_rdkit()
    return {s.decode(): mol for s, mol in zip(smiles, molecules)}


def store_as_json_array(
    geometries: List[Chem.rdchem.Mol], file_name: str = "geometries.json"
) -> None:
    """Store ``geometries`` as a JSON array."""
    array = [Chem.MolToPDBBlock(mol) for mol in geometries]
    with open(file_name, "w") as handler:
        handler.write(json.dumps(array, indent=4))


def main():
    parser = argparse.ArgumentParser("modeller")
    # configure logger
    parser.add_argument("-f", "--files", help="HDF5 files", nargs="+")
    parser.add_argument("-c", "--csv", help="CSV files with the smiles")
    args = parser.parse_args()
    extract_data(args.files, args.csv)


if __name__ == "__main__":
    main()
