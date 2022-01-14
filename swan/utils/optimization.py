import torch
from torch import nn

class SwanOptimizer:
    """Wrapper around pytorch optimizer, including a scheduler."""
    def __init__(self, name: str = 'Adam', learning_rate: int = 3e-4, weight_decay: float = 1.,
                 scheduler_name: str = None, **scheduler_kwargs):
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        
        # set by connect_optimizer, because needs the model
        self.optimizer = None  
        self.scheduler = None

    def connect_optimizer(self, model: nn.Module):
        self.optimizer = getattr(torch.optim, self.name)(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.scheduler_name is not None:
            self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(
                self.optimizer, **self.scheduler_kwargs)
        
    def step(self, loss):
        """Perform a single training step, and if using a scheduler update the learning rate.

        Parameters
        ----------
        loss
            Value of loss function to backprogagate through.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler_name:
            self.scheduler.step()
    
    def get_config(self):
        config = {
            'optimizer_name': self.name,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler_name': self.scheduler_name,
        }
        scheduler_config = {'scheduler_' + key: value for key, value in self.scheduler_kwargs.items()}
        config.update(scheduler_config)
        return config