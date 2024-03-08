import neuro.functions as function
from neuro.tensors import Tensor
from typing import Optional


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    out = function.Matmul.apply(x, w.transpose())
    if b is not None:
        out = out + b
    return out
