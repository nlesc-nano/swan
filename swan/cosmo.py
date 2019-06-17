from subprocess import (PIPE, Popen)

import argparse
import logging
import numpy as np
import os
import pandas as pd

# Starting logger
logger = logging.getLogger(__name__)


def main():
    # configure logger
    config_logger(".")

    parser = argparse.ArgumentParser(description="cosmos -i file_smiles")
    parser.add_argument('-i', required=True,
                        help="Input file in with the smiles")
    parser.add_argument('-s', help='solvent', default="CC1=CC=CC=C1")
    args = parser.parse_args()

    inp = {"file_smiles": args.i, "solvent": args.s}

    # compute_activity_coefficient
    df = compute_activity_coefficient(inp)

    df.to_csv("Gammas.csv", sep='\t')


def compute_activity_coefficient(opt: dict) -> pd.DataFrame:
    """
    Call the Unicaf method from ADf-Cosmo to compute the activation coefficient:
    https://www.scm.com/doc/COSMO-RS/UNIFAC_program/Input_formatting.html?highlight=smiles
    """
    smiles = np.loadtxt(opt["file_smiles"], dtype=str)
    gammas = np.empty_like(smiles)

    for i, x in np.ndenumerate(smiles):
        gammas[i] = call_unicaf(opt, x)
        
    return pd.DataFrame(data=gammas, index=smiles, columns=['gamma'])


def call_unicaf(opt: dict, smile: str) -> float:
    """
    Call the Unicaf executable from ADF
    """
    cmd = f'unifac -smiles {opt["solvent"]} "{smile}" -x 1 0  -t ACTIVITYCOEF'.split('\n')
    rs = run_command(cmd)

    if rs[1]:
        # There was an error
        return np.nan
    else:
        return read_gamma(rs[0])


def read_gamma(xs: bytes) -> float:
    """
    Read the gamma value (activity coefficient) from the Unicaf output
    """
    arr = [x.lstrip() for x in xs.split(b'\n') if x]
    index = arr.index(b'gamma') + 2

    return float(arr[index])


def run_command(cmd: str, workdir: str="."):
    """
    Run a bash command using subprocess
    """
    with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, cwd=workdir) as p:
        rs = p.communicate()

    logger.info("RUNNING COMMAND: {}".format(cmd))
    logger.error("COMMAND ERROR: {}".format(rs[1].decode()))

    return rs


def config_logger(workdir: str):
    """
    Setup the logging infrasctucture.
    """
    file_log = os.path.join(workdir, 'output.log')
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(asctime)s---%(levelname)s\n%(message)s\n',
                        datefmt='[%I:%M:%S]')
    logging.getLogger("noodles").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.terminator = ""
