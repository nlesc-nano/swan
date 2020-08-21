"""Interface with CAT/PLAMS Packages."""
import logging
import os
import shutil
import tempfile
from contextlib import redirect_stderr
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Mapping, Tuple, TypeVar

import h5py
import numpy as np
import pandas as pd
import scm.plams.interfaces.molecule.rdkit as molkit
import yaml
from more_itertools import chunked
from scm.plams import CRSJob, Settings
from retry import retry

import CAT
from CAT.base import prep
from dataCAT import prop_to_dataframe
from nanoCAT.ligand_solvation import get_solv

from ..utils import Options
from .functions import run_command

T = TypeVar('T')

# Starting logger
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('CAT')
logger.propagate = False
handler = logging.FileHandler("cat_output.log")
logger.addHandler(handler)


@retry(FileExistsError, tries=3, delay=1)
def call_cat(smiles: pd.Series, opts: Mapping[str, T], chunk_name: str = "0") -> Path:
    """Call cat with a given `config` and returns a dataframe with the results.

    Parameters
    ----------
    molecules
        Pandas Series with the smiles to compute
    opts
        Options for the computation
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
       bulkiness: true
    ligand:
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
    path_hdf5 = call_cat(smiles, opts, chunk_name=chunk_name)
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
    """Compute a ligand/quantum dot property using CAT."""
    worker = partial(compute_bulkiness, smiles, opts.to_dict())

    with Pool() as p:
        results = p.map(worker, chunked(smiles.index, 10))

    results = np.concatenate(results)

    print("len smiles: ", len(smiles.index), "len bulkiness: ", results.size)
    if len(smiles.index) != results.size:
        msg = "WWW There is an incongruence in the bulkiness computed by CAT!"
        raise RuntimeError(msg)

    return results


def call_mopac(smile: str, solvents=["Toluene.coskf"]) -> Tuple[float, float]:
    """Use the COsMO-RS to compute the activity coefficient.

    see https://www.scm.com/doc/COSMO-RS/Fast_Sigma_QSPR_COSMO_sigma-profiles.html
    """
    # Call fast sigma
    fast_sigma = Path(os.environ['ADFBIN']) / 'fast_sigma'
    cmd = f'{fast_sigma} --smiles "{smile}"'

    try:
        tmp = tempfile.mkdtemp(prefix="cat_")
        rs = run_command(cmd, workdir=tmp)
        if rs[1]:
            return np.nan, np.nan
        return call_cat_mopac(Path(tmp), smile, solvents)
    except ValueError:
        print(f"Error reading smile: {smile}")
        return np.nan, np.nan

    finally:
        if Path(tmp).exists():
            shutil.rmtree(tmp)


def call_cat_mopac(tmp: Path, smile: str, solvents: list):
    """Use the CAT to call MOPAC."""
    # Call COSMO
    coskf = tmp / 'CRSKF'

    mol = molkit.from_smiles(smile)
    s = Settings()
    s.update(CAT.get_template('qd.yaml')['COSMO-RS activity coefficient'])
    s.update(CAT.get_template('crs.yaml')['MOPAC PM6'])
    s.input.Compound.compkffile = ''

    # Prepare the job settings and solvent list
    coskf_path = Path(CAT.__path__[0]) / 'data' / 'coskf'
    solvents = [(coskf_path / solv).as_posix() for solv in solvents]

    # Call Cosmo
    energy_solvation, gamma = get_solv(
        mol, solvents, coskf.as_posix(), job=CRSJob, s=s)

    return tuple(map(check_output, (energy_solvation, gamma)))


def check_output(xs):
    """Check that there is a valid output in x."""
    if xs and np.isreal(xs[0]):
        return xs[0]
    return np.nan
