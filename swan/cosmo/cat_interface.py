from .functions import run_command
from nanoCAT.analysis.ligand_solvation import get_solv
from pathlib import Path
from scm.plams import Settings

import CAT
import logging
import numpy as np
import shutil
import os
import scm.plams.interfaces.molecule.rdkit as molkit
import tempfile

# Starting logger
logger = logging.getLogger(__name__)


def call_mopac(smile: str, solvents=["Toluene.coskf"]) -> float:
    """
    Use the COsMO-RS to compute the activity coefficient, see:
    https://www.scm.com/doc/COSMO-RS/Fast_Sigma_QSPR_COSMO_sigma-profiles.html
    """
    # Call fast sigma
    fast_sigma = Path(os.environ['ADFBIN']) / 'fast_sigma'
    cmd = f'{fast_sigma} --smiles "{smile}"'

    try:
        tmp = tempfile.mkdtemp(prefix="cat_")
        rs = run_command(cmd, workdir=tmp)
        if rs[1]:
            return np.nan, np.nan
        else:
            return call_cat_mopac(Path(tmp), smile, solvents)
    except ValueError:
        logger.warning(f"Error reading smile: {smile}")
        return np.nan, np.nan

    finally:
        if Path(tmp).exists():
            shutil.rmtree(tmp)


def call_cat_mopac(tmp: Path, smile: str, solvents: list):
    """
    use the CAT to call MOPAC.
    """
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
        mol, solvents, coskf.as_posix(), job=CAT.CRSJob, s=s, keep_files=False)

    return tuple(map(check_output, (E_solv, gamma)))


def check_output(xs):
    """
    Check that there is a valid output in x.
    """
    if xs and np.isreal(xs[0]):
        return xs[0]
    else:
        return np.nan
