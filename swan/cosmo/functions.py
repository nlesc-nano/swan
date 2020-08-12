from subprocess import (PIPE, Popen)
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple

# Starting logger
logger = logging.getLogger(__name__)


def run_command(cmd: str, workdir: str = ".") -> Tuple[bytes, bytes]:
    """
    Run a bash command using subprocess
    """
    with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, cwd=workdir) as p:
        rs = p.communicate()

    if rs[1]:
        logger.info("RUNNING COMMAND: {}".format(cmd))
        logger.error("COMMAND ERROR: {}".format(rs[1].decode()))

    return rs


def merge_csv(path: str, output: str, patt: str = "Gamma*.csv") -> pd.DataFrame:
    """
    Read all the csv files from a given `path` and generates a single dataframe
    """
    p = Path(path)
    files = [x.as_posix() for x in p.rglob(patt)]
    df = pd.read_csv(files[0])

    # read all the csv files
    for f in files[1:]:
        s = pd.read_csv(f)
        df = pd.concat((df, s))

    df.to_csv(output)

    return df
