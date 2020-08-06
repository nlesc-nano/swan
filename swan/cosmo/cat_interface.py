"""Interface with CAT/PLAMS Packages."""
from pathlib import Path
import logging
import shutil
import os
import tempfile
import numpy as np
import pandas as pd
import CAT
import yaml
from CAT.base import prep
from nanoCAT.ligand_solvation import get_solv
from scm.plams import (CRSJob, Settings)
import scm.plams.interfaces.molecule.rdkit as molkit
from .functions import run_command
from ..utils import Options

# Starting logger
LOGGER = logging.getLogger(__name__)


def call_cat(molecules: pd.DataFrame, opts: Options) -> Path:
    """Call cat with a given `config` and returns a dataframe with the results.

    Parameters
    ----------
    molecules
        Dataframe with the molecules to compute
    opts
        Options for the computation

    Returns
    -------
    Path to the HDF5 file with the results

    Raises
    ------
    RuntimeError
        If the Cat calculation fails
    """
    # create workdir for cat
    path_workdir_cat = Path(opts.workdir) / "cat_workdir"
    path_workdir_cat.mkdir()

    path_smiles = (Path(opts.workdir) / "smiles.csv").absolute().as_posix()

    # Save smiles of the candidates
    molecules.to_csv(path_smiles, columns=["smiles"], index=False, header=False)

    input_cat = yaml.load(f"""
path: {path_workdir_cat.absolute().as_posix()}

input_cores:
    - {opts.core}:
        guess_bonds: False

input_ligands:
    - {path_smiles}

optional:
    qd:
       bulkiness: true
""", Loader=yaml.FullLoader)

    inp = Settings(input_cat)
    prep(inp)

    path_hdf5 = path_workdir_cat / "database" / "structures.hdf5"

    if not path_hdf5.exists():
        raise RuntimeError(f"There is not hdf5 file at:{path_hdf5}")
    else:
        return path_hdf5


def call_mopac(smile: str, solvents=["Toluene.coskf"]) -> float:
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
        LOGGER.warning(f"Error reading smile: {smile}")
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
    E_solv, gamma = get_solv(
        mol, solvents, coskf.as_posix(), job=CRSJob, s=s, keep_files=False)

    return tuple(map(check_output, (E_solv, gamma)))


def check_output(xs):
    """Check that there is a valid output in x."""
    if xs and np.isreal(xs[0]):
        return xs[0]
    return np.nan
