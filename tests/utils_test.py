"""Functions use for testing."""

from pathlib import Path

import pkg_resources as pkg

__all__ = ["PATH_SWAN", "PATH_TEST"]

# Environment data
PATH_SWAN = Path(pkg.resource_filename('swan', ''))
ROOT = PATH_SWAN.parent

PATH_TEST = ROOT / "tests" / "test_files"
