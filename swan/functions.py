from subprocess import (PIPE, Popen)
import logging

# Starting logger
logger = logging.getLogger(__name__)


def run_command(cmd: str, workdir: str = "."):
    """
    Run a bash command using subprocess
    """
    with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, cwd=workdir) as p:
        rs = p.communicate()

    logger.info("RUNNING COMMAND: {}".format(cmd))
    if rs[1]:
        logger.error("COMMAND ERROR: {}".format(rs[1].decode()))

    return rs


def chunks_of(xs: list, n: int):
    """Yield successive n-sized chunks from xs"""
    for i in range(0, len(xs), n):
        yield xs[i:i + n]
