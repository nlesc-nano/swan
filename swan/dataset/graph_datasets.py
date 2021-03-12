"""Module to process dataset."""
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from rdkit.Chem import PandasTools
from .graph.molecular_graph import create_molecular_graph_data
from .sanitize_data import sanitize_data


class MolGraphDataset(tg.data.Dataset):
    """Dataset for molecular graphs."""
    def __init__(self,
                 root: str,
                 data: Union[pd.DataFrame, str],
                 properties: List[str] = None,
                 sanitize=True):
        """Generate a dataset using graphs

        Args:
            root (str): [description]
            data (Union[pd.DataFrame, str]): path of the csv file or pd DF
            properties (List[str], optional): Names of the properies to use as label.
                                              Defaults to None.
        """
        super().__init__(root)

        # convert to pd dataFrame if necessary
        if isinstance(data, str):
            data = pd.read_csv(data)
            PandasTools.AddMoleculeColumnToFrame(data,
                                                 smilesCol='smiles',
                                                 molCol='molecules')
        if sanitize:
            data = sanitize_data(data)

        data.reset_index(drop=True, inplace=True)

        # extract molecules and positions
        self.molecules = data['molecules']
        self.positions = data[
            'positions'] if "positions" in data.columns else None

        self.norm = tg.transforms.NormalizeFeatures()

        # get labels
        if properties is not None:
            self.labels = data[properties].to_numpy(np.float32)
        else:
            self.labels = None

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self):
        """Return dataset length."""
        return len(self.molecules)

    def __getitem__(self, idx):
        """Return the idx dataset element."""
        labels = None if self.labels is None else torch.Tensor(
            [self.labels[idx]])
        positions = None if self.positions is None else torch.Tensor(
            self.positions[idx])
        data = create_molecular_graph_data(self.molecules[idx],
                                           positions=positions,
                                           labels=labels)
        return self.norm(data)
