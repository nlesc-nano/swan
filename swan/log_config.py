from pathlib import Path
import logging
import sys


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

    # capture stdout and stderr
    log = logging.getLogger(__name__)
    sys.stdout = LoggerWriter(log.info)
    sys.stderr = LoggerWriter(log.warning)


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        self.level(sys.stderr)
