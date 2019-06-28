from functools import partial
from multiprocessing import Pool
from pathlib import Path
from scm.plams import init, finish
from swan.log_config import config_logger
from swan.cosmo.cat_interface import call_mopac
from swan.cosmo.functions import (chunks_of, run_command)
from swan.utils import Options

import argparse
import numpy as np
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="cosmos -i file_smiles")
    parser.add_argument('-i', required=True,
                        help="Input file in with the smiles")
    parser.add_argument('-s', help='solvent', default="CC1=CC=CC=C1")
    parser.add_argument(
        '-n', help='Number of molecules per file', default=10000, type=int)
    parser.add_argument(
        '-p', help='Number of processes', default=1, type=int)
    parser.add_argument('-w', help="workdir", default=Path("."))
    parser.add_argument('-csv', help="Current csv data", default=None)

    args = parser.parse_args()

    if args.csv is not None:
        df = pd.read_csv(args.csv, sep='\t', index_col=0)
    else:
        df = pd.DataFrame(columns=["E_solv", "gammas"])

    inp = {"file_smiles": args.i, "solvent": args.s, "size_chunk": args.n, "workdir": args.w,
           "processes": args.p, "data": df}

    # configure logger
    config_logger(args.w)

    # compute_activity_coefficient
    init()
    compute_activity_coefficient(Options(inp))
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

    # split the smiles into chunks and run it in multiple processes
    size = opt.size_chunk

    fun = partial(call_cosmo_on_chunk, opt.data)
    with Pool(processes=opt.processes) as p:
        files = p.starmap(fun, enumerate(chunks_of(smiles, size)))

    return files


def call_cosmo_on_chunk(data: pd.DataFrame, k: int, smiles: list) -> str:
    """
    Call chunk `k` containing the list of string given by `smiles`
    """
    df = pd.DataFrame(columns=data.columns)

    for i, x in enumerate(smiles):
        if x in data.index:
            df.loc[x] = data.loc[x]
        else:
            df.loc[x] = call_mopac(x)

    # Store the chunk in a file
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
