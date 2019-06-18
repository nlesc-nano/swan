from .function import run_command
from CAT.analysis.ligand_solvation import get_solv
from os.path import join
from scm.plams import init, finish, Settings

import os
import scm.plams.interfaces.molecule.rdkit as molkit
import CAT


def call_fast_sigma():
    """
    https://www.scm.com/doc/COSMO-RS/Fast_Sigma_QSPR_COSMO_sigma-profiles.html
    """
    smiles = 'OS(=O)(=O)CCC=C'
    fast_sigma = join(os.environ['ADFBIN'], 'fast_sigma')
    cmd = " ".join([fast_sigma, '--smiles', smiles])
    run_command(cmd)
    coskf = join(os.getcwd(), 'CRSKF')

    mol = molkit.from_smiles(smiles)
    s = Settings()
    s.update(CAT.get_template('qd.yaml')['COSMO-RS activity coefficient'])
    s.update(CAT.get_template('crs.yaml')['MOPAC PM6'])
    s.input.Compound.compkffile = ''

    # Prepare the job settings and solvent list
    coskf_path = join(CAT.__path__[0], 'data', 'coskf')
    solvent_list = sorted([join(coskf_path, solv) for solv in os.listdir(coskf_path) if
                           solv not in ('__init__.py', 'README.rst')])

    init()
    E_solv, gamma = get_solv(mol, solvent_list, coskf, job=CAT.CRSJob, s=s)
    finish()
