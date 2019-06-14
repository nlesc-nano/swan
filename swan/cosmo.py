from subprocess import (PIPE, Popen)
import logging
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


def Compute_activity_coefficient(inp: dict) -> pd.DataFrame:
    """
    Call the Unicaf method from ADf-Cosmo: https://www.scm.com/doc/COSMO-RS/UNIFAC_program/Input_formatting.html?highlight=smiles
    to compute the activation coefficient
    """
    pass


def run_command(cmd: str, workdir: str):
    """
    Run a bash command using subprocess
    """
    with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, cwd=workdir.as_posix()) as p:
        rs = p.communicate()

    logger.info("RUNNING COMMAND: {}".format(cmd))
    logger.info("COMMAND OUTPUT: {}".format(rs[0].decode()))
    logger.error("COMMAND ERROR: {}".format(rs[1].decode()))

    return 42


def config_logger(workdir: str):
    """
    Setup the logging infrasctucture.
    """
    file_log = os.path.join(workdir, 'xtp.log')
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(asctime)s---%(levelname)s\n%(message)s\n',
                        datefmt='[%I:%M:%S]')
    logging.getLogger("noodles").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.terminator = ""
