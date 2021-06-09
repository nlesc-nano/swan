from typing import Tuple, TypeVar, Union, NamedTuple, Generic

import numpy as np
import torch

T_co = TypeVar('T_co', bound=Union[np.ndarray, torch.Tensor], covariant=True)


class SplitDataset(NamedTuple, Generic[T_co]):
    indices: np.ndarray       # Shuffled indices to split the data
    ntrain: int               # Number of points used for training
    features_trainset: T_co   # Features for training
    features_validset: T_co   # Features for validation
    labels_trainset: T_co     # Labels for training
    labels_validset: T_co     # Labels for validation


def split_dataset(features: T_co, labels: T_co, frac: Tuple[float, float] = (0.8, 0.2)) -> SplitDataset:
        """Split the fingerprint dataset into a training and validation set.

        Parameters
        ----------
        features
            Dataset features
        labels
            Dataset labels
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        # Generate random indices to train and validate the model
        size = len(features)
        indices = np.arange(size)
        np.random.shuffle(indices)

        ntrain = int(size * frac[0])
        features_trainset = features[indices[:ntrain]]
        features_validset = features[indices[ntrain:]]
        labels_trainset = labels[indices[:ntrain]]
        labels_validset = labels[indices[ntrain:]]

        return SplitDataset(indices, ntrain, features_trainset, features_validset, labels_trainset, labels_validset)
