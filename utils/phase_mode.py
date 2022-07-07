import torch.nn as nn
from typing import Any


class Training_phase(object):
    r"""
    model:nn.Module
    >>> model.training = True
    >>> with Training_phase(model, mode=False):
    ...     do_val ...
    >>> model.training
    True
    """
    def __init__(self, model:nn.Module, mode:bool):
        self.prev_mode:bool = model.training
        self.model = model
        self.model.train(model)
        
        self.prev_mode:bool = model.training
        return self
    
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.model.train(self.prev_mode)
    