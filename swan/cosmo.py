from .functions import (chunks_of, run_command)
from pathlib import Path
from scm.plams import init, finish

import argparse
import logging
import numpy as np
import os
import pandas as pd

# Starting logger
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="cosmos -i file_smiles")
    parser.add_argument('-i', required=True,
                        help="Input file in with the smiles")
    parser.add_argument('-s', help='solvent', default="CC1=CC=CC=C1")
    parser.add_argument(
        '-n', help='Number of molecules per file', default=10000)
    parser.add_argument('-w', help="workdir", default=Path("."))
    args = parser.parse_args()

    inp = {"file_smiles": args.i, "solvent": args.s, "size_chunk": args.n, "workdir": args.w}

    # configure logger
    config_logger(args.w)

    # compute_activity_coefficient
    compute_activity_coefficient(inp)
    finish()


def compute_activity_coefficient(opt: dict):
    """
    Call the Unifac method from ADf-Cosmo to compute the activation coefficient:
    https://www.scm.com/doc/COSMO-RS/UNIFAC_program/Input_formatting.html?highlight=smiles
    """
    # Read the file containing the smiles
    df = pd.read_csv(opt["file_smiles"], sep="\t", header=None)
    df.rename(columns={0: "smiles"}, inplace=True)
    smiles = df["smiles"].values

    size = opt["size_chunk"]

    for k, xs in enumerate(chunks_of(smiles, size)):
        gammas = np.empty(len(xs))
        for i, x in enumerate(xs):
            gammas[i] = call_mopac(x)

        df = pd.DataFrame(data=gammas, index=xs, columns=['gamma'])
        name = f"Gammas_{k}.csv"
        df.to_csv(name, sep='\t')


def call_unifac(opt: dict, smile: str) -> float:
    """
    Call the Unifac executable from ADF
    """
    unifac = Path(os.environ['ADFBIN']) / 'unifac'
    cmd = f'{unifac} -smiles {opt["solvent"]} "{smile}" -x 1 0  -t ACTIVITYCOEF'.split(
        '\n')
    rs = run_command(cmd)

    if rs[1]:
        # There was an error
        return np.call_mopac(opt, smile)
    else:
        return read_gamma(rs[0])


def read_gamma(xs: bytes) -> float:
    """
    Read the gamma value (activity coefficient) from the Unifac output
    """
    arr = [x.lstrip() for x in xs.split(b'\n') if x]
    if b'gamma' in arr:
        index = arr.index(b'gamma') + 2
        return float(arr[index])
    else:
        return np.nan


def config_logger(workdir: Path):
    """
    Setup the logging infrasctucture.
    """
    file_log = workdir / 'output.log'
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(asctime)s---%(levelname)s\n%(message)s',
                        datefmt='[%I:%M:%S]')
    logging.getLogger("noodles").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.terminator = ""
