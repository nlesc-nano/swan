from .functions import run_command
from CAT.analysis.ligand_solvation import get_solv
from pathlib import Path
from scm.plams import init, finish, Settings

import CAT
import numpy as np
import shutil
import os
import scm.plams.interfaces.molecule.rdkit as molkit
import tempfile


def call_mopac(smile: str, solvents=["Toluene.coskf"]) -> float:
    """
    Use the COsMO-RS to compute the activity coefficient, see:
    https://www.scm.com/doc/COSMO-RS/Fast_Sigma_QSPR_COSMO_sigma-profiles.html
    """
    # Call fast sigma
    fast_sigma = Path(os.environ['ADFBIN']) / 'fast_sigma'
    cmd = f'{fast_sigma} --smiles "{smile}"'

    try:
        rs = run_command(cmd, workdir=os.getcwd())
if rs[1]:
            return np.nan
        else:
            x = call_cat_mopac(smile, solvents)
            result = x if x is not None else np.nan
            return result
    finally:
        shutil.rmtree("plams_workdir")

def call_cat_mopac(smile: str, solvents: list):
    """
    use the CAT to call MOPAC.
    """
    # Call COSMO
    coskf = Path(os.getcwd()) / 'CRSKF'

    mol = molkit.from_smiles(smile)
    s = Settings()
    s.update(CAT.get_template('qd.yaml')['COSMO-RS activity coefficient'])
    s.update(CAT.get_template('crs.yaml')['MOPAC PM6'])
    s.input.Compound.compkffile = ''

    # Prepare the job settings and solvent list
    coskf_path = Path(CAT.__path__[0]) / 'data' / 'coskf'
    solvents = [(coskf_path / solv).as_posix() for solv in solvents]

    # Call Cosmo
    init()
    E_solv, gamma = get_solv(mol, solvents, coskf.as_posix(), job=CAT.CRSJob, s=s)
    finish()

   
    return gamma
   
