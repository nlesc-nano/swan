"""Interface with CAT/PLAMS Packages.

Index
-----
.. currentmodule:: swan.cat_interface

API
---

.. autofunction:: compute_bulkiness
"""
import logging
from collections import defaultdict
from contextlib import redirect_stderr
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import (Callable, DefaultDict, List, Mapping, NamedTuple, TypeVar,
                    Union)

import h5py
import numpy as np
import pandas as pd
import yaml
from CAT.base import prep
from dataCAT import prop_to_dataframe
from more_itertools import chunked
from retry import retry
from scm.plams import Settings

from .utils import Options

__all__ = ["compute_bulkiness"]

# Long types
T = TypeVar('T')
BatchResult = Union[np.ndarray, pd.DataFrame]
Callback = Callable[[pd.Series, Mapping[str, T], pd.Index], BatchResult]
Reducer = Callable[[List[BatchResult]], BatchResult]

# Starting logger
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('CAT')
logger.propagate = False
handler = logging.FileHandler("cat_output.log")
logger.addHandler(handler)


class PropertyMetadata(NamedTuple):
    """Name and metadata of the property computed by CAT."""

    name: str
    dset: str  # Dset in the HDF5


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


def compute_property_using_cat(
        smiles: pd.Series, opts: Mapping[str, T],
        chunk_name: str, metadata: PropertyMetadata) -> pd.Series:
    """Compute the bulkiness for the candidates."""
    # Properties to compute using cat
    cat_properties = defaultdict(bool)
    cat_properties[metadata.name] = True

    # run cat
    path_hdf5 = call_cat(smiles, opts, cat_properties, chunk_name=chunk_name)
    with h5py.File(path_hdf5, 'r') as f:
        dset = f[metadata.dset]
        df = prop_to_dataframe(dset)

    # flat the dataframe and remove duplicates
    df.reset_index(inplace=True)

    # make anchor atom neutral to compare with the original
    # TODO make it more general
    df.ligand = df.ligand.str.replace("[O-]", "O", regex=False)

    # remove duplicates
    df.drop_duplicates(subset=['ligand'], keep='first', inplace=True)

    return df


def compute_batch_bulkiness(smiles: pd.Series, opts: Mapping[str, T], indices: pd.Index) -> pd.Series:
    """Compute bulkiness using CAT."""
    chunk = smiles[indices]
    chunk_name = str(indices[0])

    # compute and extract the bulkiness
    metadata = PropertyMetadata("bulkiness", 'qd/properties/V_bulk')
    df = compute_property_using_cat(chunk, opts, chunk_name, metadata)

    bulkiness = pd.merge(chunk, df, left_on="smiles", right_on="ligand")["V_bulk"]
    if len(smiles.index) != len(bulkiness):
        msg = "There is an incongruence in the bulkiness computed by CAT!"
        logger.error(f"There was an error processing chunk: {chunk_name}\n{msg}")
        values = np.repeat(np.nan, len(indices))
    else:
        values = bulkiness.to_numpy()
    return values


def compute_batch_cosmo_rs(smiles: pd.Series, opts: Mapping[str, T], indices: pd.Index) -> pd.Series:
    """Compute the cosmo_rs properties of the `smiles` with `indices`."""
    # chunk = smiles[indices]
    # chunk_name = str(indices[0])

    # # compute and extract the bulkiness
    # metadata = PropertyMetadata("gamma", 'qd/properties/V_bulk')
    # df = compute_property_using_cat(chunk, opts, chunk_name, metadata)

    raise NotImplementedError


def map_reduce(smiles: pd.Series, opts: Options,
               callback: Callback, reduce: Reducer) -> BatchResult:
    """Distribute the properties computation in batches."""
    worker = partial(callback, smiles, opts.to_dict())

    with Pool() as p:
        results = p.map(worker, chunked(smiles.index, 10))

    return reduce(results)


def compute_bulkiness(smiles: pd.Series, opts: Options) -> np.ndarray:
    """Compute a ligand/quantum dot bulkiness using CAT.

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
    results = map_reduce(smiles, opts, compute_batch_bulkiness, np.concatenate)

    if len(smiles.index) != results.size:
        msg = "There is an incongruence in the bulkiness computed by CAT!"
        raise RuntimeError(msg)

    return results


def compute_cosmo_rs(smiles: pd.Series, opts: Options) -> pd.DataFrame:
    """Compute bulkiness using CAT.

    It creates several instances of CAT using multiprocessing.

    Parameters
    ----------
    smiles
        Pandas.Series with the smiles to compute
    opts
        Options to call CAT

    Returns
    -------
        DataFrame with the properties values

    """
    results = map_reduce(smiles, opts, compute_batch_cosmo_rs, pd.concat)

    return results
