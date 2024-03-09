import neuro.functions as fn
from neuro.tensors import Tensor
from typing import Optional


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    out = fn.Matmul.apply(x, w.transpose())
    if b is not None:
        out = out + b
    return out


def sigmoid(z: Tensor):
    nz = fn.Neg.apply(z)
    out = 1 / (1 + fn.Exp.apply(nz))
    return out

def tanh(z: Tensor):
    pass
