from functools import partial
from multiprocessing import Pool
from subprocess import (PIPE, Popen)

import argparse
import logging
import numpy as np
import os
import pandas as pd

# Starting logger
logger = logging.getLogger(__name__)


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


def main():
    # configure logger
    config_logger(".")

    parser = argparse.ArgumentParser(description="cosmos -i file_smiles")
    parser.add_argument('-i', required=True,
                        help="Input file in with the smiles")
    parser.add_argument('-s', help='solvent', default="CC1=CC=CC=C1")
    args = parser.parse_args()

    inp = {"file_smiles": args.i, "solvent": args.s}

    # # input arguments
    opt = Options(inp)
    print(opt)

    # compute_activity_coefficient
    df = compute_activity_coefficient(opt)

    print(df.head())


def compute_activity_coefficient(opt: dict) -> pd.DataFrame:
    """
    Call the Unicaf method from ADf-Cosmo to compute the activation coefficient:
    https://www.scm.com/doc/COSMO-RS/UNIFAC_program/Input_formatting.html?highlight=smiles
    """
    fun = partial(call_unicaf, opt)
    smiles = np.loadtxt(opt.file_smiles, dtype=str)
    with Pool() as p:
        gammas = p.map(fun, smiles)

    return pd.DataFrame(data=gammas, index=smiles, columns=['gamma'])


def call_unicaf(opt: dict, smile: str) -> float:
    """
    Call the Unicaf executable from ADF
    """
    cmd = f'unifac -smiles {opt.solvent} {smile} -x 1 0  -t ACTIVITYCOEF'
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


def run_command(cmd: str, workdir: str):
    """
    Run a bash command using subprocess
    """
    with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, cwd=workdir.as_posix()) as p:
        rs = p.communicate()

    logger.info("RUNNING COMMAND: {}".format(cmd))
    logger.info("COMMAND OUTPUT: {}".format(rs[0].decode()))
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
