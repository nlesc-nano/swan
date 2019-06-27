from pathlib import Path
from swan.log_config import (config_logger, LoggerWriter)
import logging
import sys


def test_logger(tmp_path, caplog):
    workdir = Path(tmp_path)
    config_logger(workdir)

    # Log some stuff
    log = logging.getLogger()
    log.info("UP AND RUNNING!!")

    # Capture the standard output/error
    sys.stdout = LoggerWriter(log.info)
    sys.stderr = LoggerWriter(log.warning)

    print("The answer is 42")
