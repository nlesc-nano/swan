"""Functions use for testing."""

from pathlib import Path
import os
import pkg_resources as pkg

__all__ = ["PATH_SWAN", "PATH_TEST"]

# Environment data
PATH_SWAN = Path(pkg.resource_filename('swan', ''))
ROOT = PATH_SWAN.parent

PATH_TEST = ROOT / "tests" / "files"


def remove_files():
    """Remove files used to train the models."""
    files = ["swan_state.h5", "swan_output.log"]
    for f in files:
        if Path(f).exists():
            os.remove(f)
