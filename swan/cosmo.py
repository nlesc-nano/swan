from .cat_interface import call_mopac
from .functions import (chunks_of, run_command)
from pathlib import Path
from scm.plams import init, finish
from multiprocessing import Pool

import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys

# Starting logger
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="cosmos -i file_smiles")
    parser.add_argument('-i', required=True,
                        help="Input file in with the smiles")
    parser.add_argument('-s', help='solvent', default="CC1=CC=CC=C1")
    parser.add_argument(
        '-n', help='Number of molecules per file', default=10000, type=int)
    parser.add_argument(
        '-p', help='Number of processes', default=10, type=int)
    parser.add_argument('-w', help="workdir", default=Path("."))
    args = parser.parse_args()

    inp = {"file_smiles": args.i, "solvent": args.s, "size_chunk": args.n, "workdir": args.w,
           "processes": args.p}

    # configure logger
    config_logger(args.w)

    # compute_activity_coefficient
    init()
    compute_activity_coefficient(inp)
    finish()


def compute_activity_coefficient(opt: dict):
    """
    Call the ADf-Cosmo method to compute the activation coefficient:
    https://www.scm.com/doc/COSMO-RS/UNIFAC_program/Input_formatting.html?highlight=smiles
    """
    # Read the file containing the smiles
    df = pd.read_csv(opt["file_smiles"], sep="\t", header=None)
    df.rename(columns={0: "smiles"}, inplace=True)
    smiles = df["smiles"].values

    size = opt["size_chunk"]

    with Pool(processes=4) as p:
        files = p.starmap(call_cosmo_on_chunk,
                          enumerate(chunks_of(smiles, size)))

    return files


def call_cosmo_on_chunk(k: int, smiles: list) -> str:
    """
    Call chunk `k` containing the list of string given by `smiles`
    """
    gammas = np.empty(len(smiles))
    E_solv = np.empty(len(smiles))
    for i, x in enumerate(smiles):
        x, y = call_mopac(x)
        E_solv[i] = x
        gammas[i] = y

    df = pd.DataFrame(data={"gammas": gammas, "E_solv": E_solv}, index=smiles)
    name = f"Gammas_{k}.csv"
    df.to_csv(name, sep='\t')

    return name


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
    logging.getLogger("command").setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.terminator = ""
