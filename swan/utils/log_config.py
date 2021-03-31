"""Logger configuration."""

import logging
from pathlib import Path

import pkg_resources

__all__ = ["configure_logger"]

logger = logging.getLogger(__name__)


def configure_logger(workdir: Path, package_name: str = "swan"):
    """Set the logging infrasctucture."""
    file_log = workdir / 'swan_output.log'
    logging.basicConfig(filename=file_log, level=logging.INFO,
                        format='%(asctime)s  %(message)s',
                        datefmt='[%I:%M:%S]')
    handler = logging.StreamHandler()
    handler.terminator = ""

    version = pkg_resources.get_distribution(package_name).version
    path = pkg_resources.resource_filename(package_name, '')

    logger.info(f"Using {package_name} version: {version}\n")
    logger.info(f"{package_name} path is: {path}\n")
    logger.info(f"Working directory is: {workdir}")
