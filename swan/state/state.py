"""Module to interact with HDF5."""

from pathlib import Path
from typing import Any, List, Optional, Union

import h5py
import numpy as np

from swan.type_hints import PathLike, ArrayLike


class StateH5:
    """Class to interact with a HDF5 file storing the training step."""

    def __init__(self, path_hdf5: Optional[PathLike] = None, replace_state: bool = False) -> None:
        self.path = "swan_state.h5" if path_hdf5 is None else path_hdf5
        self.path = Path(self.path)

        if replace_state and self.path.exists():
            self.path.unlink()

        if not self.path.exists():
            self.path.touch()

    def has_data(self, data: Union[str, List[str]]) -> bool:
        """Search if the node exists in the HDF5 file.

        Parameters
        ----------
        data
            either Node path or a list of paths to the stored data

        Returns
        -------
        Whether the data is stored or not
        """
        with h5py.File(self.path, 'r+') as f5:
            if isinstance(data, list):
                return all(path in f5 for path in data)
            else:
                return data in f5

    def store_array(self, node: str, data: ArrayLike, dtype: str = "float") -> None:
        """
        Store a tensor as a dataset in the HDF5 file. 
        If the dataset already exists, it overwrites it.

        Parameters
        ----------
        node
            node where the data is going to be stored
        data
            Numpy array or list of array to store
        dtype
            dtype of data
        """
        supported_types = {'int': int, 'float': float, 'str': h5py.string_dtype(encoding='utf-8')}
        if dtype in supported_types:
            dtype = supported_types[dtype]
        else:
            msg = f"It is not possible to store data using type: {dtype}"
            raise RuntimeError(msg)

        with h5py.File(self.path, 'r+') as f5:
            if node in f5:
                del f5[node]
            f5.create_dataset(node, shape=np.shape(data), data=data, dtype=dtype)

    def retrieve_data(self, paths_to_prop: str) -> Any:
        """Read Numerical properties from ``paths_hdf5``.

        Parameters
        ----------
        path_to_prop
            str

        Returns
        -------
        Numpy array, float or int

        Raises
        ------
        RuntimeError
            The property has not been found
        """
        try:
            with h5py.File(self.path, 'r+') as f5:
                return f5[paths_to_prop][()]
        except KeyError:
            msg = f"There is not {paths_to_prop} stored in the HDF5\n"
            raise KeyError(msg)

    def show(self):
        """Print the keys store in the HDF5."""
        with h5py.File(self.path, 'r') as f5:
            keys = list(f5.keys())

        data = "\n".join(keys)
        print("Available data stored in the state file:\n", data)
