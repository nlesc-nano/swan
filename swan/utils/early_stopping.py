import logging
from typing import Callable

import numpy as np

# Starting logger
LOGGER = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 7, delta: float = 1e-6):
        """
        Parameters
        -----------
        patience
            How long to wait after last time validation loss improved.
        delta
            Minimum change in the monitored quantity to qualify as an improvement.

        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, saver: Callable, epoch: int, val_loss: float) -> None:
        """Update the early stopping state.

        Parameters
        ----------
        saver
            Function to save the current state of the network and the other training parameters
        epoch
            Current epoch
        val_loss
            Validation loss

        """

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(saver, epoch, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            LOGGER.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(saver, epoch, val_loss)
            self.counter = 0

    def save_checkpoint(self, saver: Callable[[int, float], None], epoch: int, val_loss: float):
        """Saves model when validation loss decrease.

        Parameters
        ----------
        saver
            Function to save the current state of the network and the other training parameters
        epoch
            Current epoch
        val_loss
            Validation loss

        """
        LOGGER.debug(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model")
        saver(epoch, val_loss)
        self.val_loss_min = val_loss
