"""Interface with CAT/PLAMS Packages.

Index
-----
.. currentmodule:: swan.cat_interface

API
---

.. autofunction:: call_cat_in_parallel
"""
import logging
from collections import defaultdict
from contextlib import redirect_stderr
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import DefaultDict, Mapping, TypeVar

import h5py
import numpy as np
import pandas as pd
import yaml
from CAT.base import prep
from more_itertools import chunked
from scm.plams import Settings

from dataCAT import prop_to_dataframe
from retry import retry

from .utils import Options

__all__ = ["call_cat_in_parallel"]


T = TypeVar('T')

# Starting logger
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('CAT')
logger.propagate = False
handler = logging.FileHandler("cat_output.log")
logger.addHandler(handler)


@retry(FileExistsError, tries=100, delay=0.01)
def call_cat(smiles: pd.Series, opts: Mapping[str, T], cat_properties: DefaultDict[str, bool],
             chunk_name: str = "0") -> Path:
    """Call cat with a given `config` and returns a dataframe with the results.

    Parameters
    ----------
    molecules
        Pandas Series with the smiles to compute
    opts
        Options for the computation
    cat_properties
        Dictionary with the name of the properties to compute
    chunk
        Name of the chunk (frame) being computed

    Returns
    -------
    Path to the HDF5 file with the results

    Raises
    ------
    RuntimeError
        If the Cat calculation fails
    """
    # create workdir for cat
    path_workdir_cat = Path(opts["workdir"]) / "cat_workdir" / chunk_name
    path_workdir_cat.mkdir(parents=True, exist_ok=True)

    path_smiles = (path_workdir_cat / "smiles.txt").absolute().as_posix()

    # Save smiles of the candidates
    smiles.to_csv(path_smiles, index=False, header=False)

    input_cat = yaml.load(f"""
path: {path_workdir_cat.absolute().as_posix()}

input_cores:
    - {opts['core']}:
        guess_bonds: False

input_ligands:
    - {path_smiles}

optional:
    qd:
       bulkiness: {cat_properties['bulkiness']}
    ligand:
       cosmo-rs: {cat_properties['cosmo-rs']}
       functional_groups:
          ['{opts["anchor"]}']
""", Loader=yaml.FullLoader)

    inp = Settings(input_cat)
    with open("cat_output.log", 'a') as f:
        with redirect_stderr(f):
            prep(inp)

    path_hdf5 = path_workdir_cat / "database" / "structures.hdf5"

    if not path_hdf5.exists():
        raise RuntimeError(f"There is not hdf5 file at:{path_hdf5}")
    else:
        return path_hdf5


def compute_bulkiness_using_cat(smiles: pd.Series, opts: Mapping[str, T], chunk_name: str) -> pd.Series:
    """Compute the bulkiness for the candidates."""
    # Properties to compute using cat
    cat_properties = defaultdict(bool)
    cat_properties["bulkiness"] = True

    # run cat
    path_hdf5 = call_cat(smiles, opts, cat_properties, chunk_name=chunk_name)
    with h5py.File(path_hdf5, 'r') as f:
        dset = f['qd/properties/V_bulk']
        df = prop_to_dataframe(dset)

    # flat the dataframe and remove duplicates
    df = df.reset_index()

    # make anchor atom neutral to compare with the original
    # TODO make it more general
    df.ligand = df.ligand.str.replace("[O-]", "O", regex=False)

    # remove duplicates
    df.drop_duplicates(subset=['ligand'], keep='first', inplace=True)

    # Extract the bulkiness
    bulkiness = pd.merge(smiles, df, left_on="smiles", right_on="ligand")["V_bulk"]

    if len(smiles.index) != len(bulkiness):
        msg = "There is an incongruence in the bulkiness computed by CAT!"
        raise RuntimeError(msg)

    return bulkiness.to_numpy()


def compute_bulkiness(smiles: pd.Series, opts: Mapping[str, T], indices: pd.Index) -> pd.Series:
    """Call CAT and catch the exceptions"""
    chunk = smiles[indices]
    chunk_name = str(indices[0])
    try:
        values = compute_bulkiness_using_cat(chunk, opts, chunk_name)
    except (RuntimeError):
        logger.error(f"There was an error processing:\n{chunk.values}")
        values = np.repeat(np.nan, len(indices))

    return values


def call_cat_in_parallel(smiles: pd.Series, opts: Options) -> np.ndarray:
    """Compute a ligand/quantum dot property using CAT.

    It creates several instances of CAT using multiprocessing.

    Parameters
    ----------
    smiles
        Pandas.Series with the smiles to compute
    opts
        Options to call CAT

    Returns
    -------
        Numpy array with the computed properties
    """
    worker = partial(compute_bulkiness, smiles, opts.to_dict())

    with Pool() as p:
        results = p.map(worker, chunked(smiles.index, 10))

    results = np.concatenate(results)

    if len(smiles.index) != results.size:
        msg = "There is an incongruence in the bulkiness computed by CAT!"
        raise RuntimeError(msg)

    return results