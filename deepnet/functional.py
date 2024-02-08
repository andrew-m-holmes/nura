import numpy as np
import deepnet.utils as utils
import deepnet.functions as fn
from deepnet.tensors import Tensor, tensor
from deepnet.types import dim
from typing import Union, Tuple, Optional


def add(a: Tensor, b: Tensor):
    a, b = atot(a, b)
    out = fn.Add.apply(a, b)
    return out


def sub(a: Tensor, b: Tensor):
    a, b = atot(a, b)
    out = fn.Sub.apply(a, b)
    return out


def mul(a: Tensor, b: Tensor):
    a, b = atot(a, b)
    out = fn.Mul.apply(a, b)
    return out


def div(a: Tensor, b: Tensor):
    a, b = atot(a, b)
    out = fn.Div.apply(a, b)
    return out


def dot(a: Tensor, b: Tensor):
    a, b = atot(a, b)
    out = fn.Dot.apply(a, b)
    return out


def matmul(a: Tensor, b: Tensor):
    a, b = atot(a, b)
    assert a.ndim >= 1 and b.ndim >= 1
    if a.ndim == 1:
        a = unsqueeze(a, 0)
    if b.ndim == 1:
        b = unsqueeze(b, -1)
    out = fn.Matmul.apply(a, b)
    return out


def pow(a: Tensor, b: Tensor):
    a, b = atot(a, b)
    out = fn.Pow.apply(a, b)
    return out


def exp(a: Tensor):
    b = atot(a)
    out = fn.Exp.apply(b)
    return out


def log(a: Tensor):
    b = atot(a)
    out = fn.Log.apply(b)
    return out


def sine(a: Tensor):
    b = atot(a)
    out = fn.Sine.apply(b)
    return out


def cosine(a: Tensor):
    b = atot(a)
    out = fn.Cosine.apply(b)
    return out


def sum(a: Tensor, dim: Optional[dim] = None, keepdims=False):
    out = fn.Sum.apply(a, dim, keepdims)
    return out


def transpose(a: Tensor, dim_0=-2, dim_1=-1):
    out = fn.Tranpose.apply(a, dim_0, dim_1)
    return out


def permute(a: Tensor, dim: Optional[dim] = None):
    out = fn.Permute.apply(a, dim)
    return out


def squeeze(a: Tensor, dim: Optional[dim] = None):
    out = fn.Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dims: Union[Tuple[int, ...], int]):
    out = fn.Unsqueeze.apply(a, dims)
    return out


def view(a: Tensor, dim: Tuple[int, ...]):
    out = fn.View.apply(a, dim)
    return out


def reshape(a: Tensor, dim: Tuple[int, ...]):
    a = tocontig(a)
    out = fn.Reshape.apply(a, dim)
    return out


def abs(a: Tensor):
    b = atot(a)
    return fn.Abs.apply(b)


def clone(a: Tensor):
    out = fn.Clone.apply(a)
    return out


def tocontig(a: Tensor):
    if utils.iscontig(a):
        return a
    b = a.clone()
    data = np.ascontiguousarray(b.data)
    return b.mutated(data=data)


def slice(a: Tensor, _slice):
    out = fn.Slice.apply(a, _slice)
    return out


def atot(*args) -> Union[Tuple[Tensor, ...], Tensor]:
    return tuple(a if utils.istensor(a) else tensor(a) for a in args)
