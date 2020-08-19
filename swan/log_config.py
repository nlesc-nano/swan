"""Logger configuration."""

from pathlib import Path
import logging
import sys

__all__ = ["configure_logger"]


def configure_logger(workdir: Path):
    """Set the logging infrasctucture."""
    file_log = workdir / 'swan_output.log'
    logging.basicConfig(filename=file_log, level=logging.INFO,
                        format='%(asctime)s  %(message)s',
                        datefmt='[%I:%M:%S]')
    handler = logging.StreamHandler(sys.stdout)
    handler.terminator = ""


class LoggerWriter:
    """Modify the default behaviour of the logger."""

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
