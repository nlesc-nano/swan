"""Utility functions."""
import h5py
from pathlib import Path
import numpy as np
from typing import Dict, TypeVar

T = TypeVar('T')


class Options(dict):
    """
    Extend the base class dictionary with a '.' notation.
    example:
    .. code-block:: python
       d = Options({'a': 1})
       d['a'] # 1
       d.a    # 1
    """

    def __init__(self, *args, **kwargs):
        """ Create a recursive Options object"""
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Options(v)

    def __getattr__(self, attr):
        """ Allow `obj.key` notation"""
        return self.get(attr)

    def __setattr__(self, key, value):
        """ Allow `obj.key = new_value` notation"""
        self.__setitem__(key, value)

    def to_dict(self) -> Dict[str, T]:
        """Convert to a normal dictionary."""
        def converter(var):
            return var.to_dict() if isinstance(var, Options) else var

        return {k: converter(v) for k, v in self.items()}


def retrieve_hdf5_data(path_hdf5: Path, paths_to_prop: str) -> np.ndarray:
    """Read Numerical properties from ``paths_hdf5``.

    Parameters
    ----------
    path_hdf5
        path to the HDF5
    path_to_prop
        str or list of str to data

    Returns
    -------
    np.ndarray
        array or list of array

    Raises
    ------
    RuntimeError
        The property has not been found

    """
    try:
        with h5py.File(path_hdf5, 'r') as f5:
            return f5[paths_to_prop][()]
    except KeyError:
        msg = f"There is not {paths_to_prop} stored in the HDF5\n"
        raise KeyError(msg)
    except OSError:
        msg = f"there is no {path_hdf5} file!"
        raise OSError(msg)
