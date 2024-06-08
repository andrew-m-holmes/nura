import numpy as np
import nura
from nura.tensors import Tensor


def computedecay(tensor: Tensor, grad: Tensor, decay: float) -> Tensor:
    return grad + tensor * decay


def xavier(w: Tensor) -> Tensor:
    raise NotImplementedError


def hamming(w: Tensor) -> Tensor:
    raise NotImplementedError
