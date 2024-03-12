import nura.functions as fn
import nura.nn.functions as nnfn
from nura.tensors import Tensor
from nura.utils import atot
from typing import Optional


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    x, w = atot(x, w)
    out = fn.Matmul.apply(x, w.transpose())
    if b is not None:
        b = atot(b)[0]
        out = out + b
    return out


def sigmoid(z: Tensor, eps=1e-6):
    z = atot(z)[0]
    nz = fn.Neg.apply(z)
    out = 1.0 / (1.0 + fn.Exp.apply(nz) + eps)
    return out


def tanh(z: Tensor, eps=1e-6):
    z = atot(z)[0]
    nz = fn.Neg.apply(z)
    e = fn.Exp.apply(z)
    ne = fn.Exp.apply(nz)
    out = (e - ne) / (e + ne + eps)
    return out


def relu(z: Tensor):
    z = atot(z)[0]
    out = nnfn.ReLU.apply(z)
    return out


def softmax(a: Tensor, pos=-1, eps=1e-6):
    a = atot(a)[0]
    e = fn.Exp.apply(a)
    out = e / (fn.Sum.apply(e, pos, keepdims=False) + eps)
    return out
