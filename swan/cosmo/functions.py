from subprocess import (PIPE, Popen)
import logging
import pandas as pd
from pathlib import Path

# Starting logger
logger = logging.getLogger(__name__)


def run_command(cmd: str, workdir: str = "."):
    """
    Run a bash command using subprocess
    """
    with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, cwd=workdir) as p:
        rs = p.communicate()

    if rs[1]:
        logger.info("RUNNING COMMAND: {}".format(cmd))
        logger.error("COMMAND ERROR: {}".format(rs[1].decode()))

    return rs


def chunks_of(xs: list, n: int):
    """Yield successive n-sized chunks from xs"""
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def merge_csv(path: str, output: str, patt: str = "Gamma*.csv") -> pd.DataFrame:
    """
    Read all the csv files from a given `path` and generates a single dataframe
    """
    p = Path(path)
    files = [x.as_posix() for x in p.rglob(patt)]
    df = pd.read_csv(files[0], sep='\t', index_col=0)

    # read all the csv files
    for f in files[1:]:
        s = pd.read_csv(f, sep='\t', index_col=0)
        df = pd.concat((df, s))

    df.to_csv(output, sep='\t')

    return df
