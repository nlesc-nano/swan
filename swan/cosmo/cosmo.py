from .cat_interface import call_mopac
from .functions import (chunks_of, run_command)
from functools import partial
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


class Options(dict):
    """
    Extend the base class dictionary with a '.' notation.
    example:
    .. code-block:: python
       d = Options({'a': 1})
       d['a'] # 1
       d.a    # 1
    """

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __deepcopy__(self, _):
        return Options(self.copy())
