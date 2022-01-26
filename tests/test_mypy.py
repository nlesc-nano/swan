"""Tests using mypy."""

import subprocess
import warnings

from typing_extensions import Literal

from tests.utils_test import PATH_SWAN, PATH_TEST

PACKAGE = PATH_SWAN.absolute().as_posix()
INI = (PATH_TEST / 'mypy.ini').absolute().as_posix()

Action = Literal['raise', 'warn', 'ignore']  #: Type annotation for the 'action' keyword.
ACTION = frozenset(['raise', 'warn', 'ignore'])


def test_mypy(action: Action = 'warn') -> None:
    """Test using mypy."""
    if action not in ACTION:
        raise ValueError(f"Invalid value for the 'action' parameter: {action!r}")

    command = f"mypy {PACKAGE!r} --config-file {INI!r}"
    out = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    stdout = out.stdout.decode()
    stderr = out.stderr.decode()
    assert not stderr, stderr

    if action == 'warn' and out.returncode != 0:
        warnings.warn(stdout)

    elif action == 'raise':
        try:
            assert out.returncode == 0, stdout
        except AssertionError as ex:
            msg = stdout.rsplit('\n', maxsplit=2)[1]
            raise AssertionError(msg) from ex
