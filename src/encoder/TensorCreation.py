from enum import Enum
import torch
from torch import nn

IDENTITY_MATRIX_ERROR = "Identity matrix must be quadratic"
MODE_ERROR = "Mode unknown"

class TensorCreationMode(Enum):
    RANDOM = 1
    ZEROS = 2
    ONES = 3
    FULL = 4
    ARANGE = 5
    UNIFORM = 6
    IDENTITY = 7

def create_tensor(mode: TensorCreationMode, d: int, n: int, fill_value=1):
    if mode == TensorCreationMode.RANDOM:
        return torch.randn(d, n)
    elif mode == TensorCreationMode.ZEROS:
        return torch.zeros(d, n)
    elif mode == TensorCreationMode.ONES:
        return torch.ones(d, n)
    elif mode == TensorCreationMode.FULL:
        return torch.full((d, n), fill_value)
    elif mode == TensorCreationMode.ARANGE:
        return torch.arange(1, d * n + 1).reshape(d, n)
    elif mode == TensorCreationMode.UNIFORM:
        return nn.Parameter(torch.rand(d, n))
    elif mode == TensorCreationMode.IDENTITY:
        if d != n:
            raise ValueError(IDENTITY_MATRIX_ERROR)
        return torch.eye(d)
    else:
        raise ValueError(MODE_ERROR)

def create_linear(input_dim, output_dim):
    return torch.nn.Linear(input_dim, output_dim)