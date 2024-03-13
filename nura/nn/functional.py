import nura.functional as f
import nura.nn.functions as fn
from nura.tensors import Tensor
from nura.utils import atot
from typing import Optional


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    out = f.matmul(x, w.transpose())
    if b is not None:
        out = out + b
    return out


def sigmoid(z: Tensor):
    out = 1.0 / (1.0 + f.exp(-z))
    return out


def tanh(z: Tensor):
    e = f.exp(z)
    ne = f.exp(-z)
    out = (e - ne) / (e + ne)
    return out


def relu(z: Tensor):
    z = atot(z)[0]
    out = fn._ReLU.apply(z)
    return out


def softmax(a: Tensor, dim=-1):
    e = f.exp(a)
    out = e / (e.sum(dim, keepdims=False))
    return out
