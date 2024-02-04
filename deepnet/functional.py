import numpy as np
import deepnet.utils as utils
import deepnet.functions as fn
from deepnet.tensors import Tensor, tensor
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
    a = atot(a)
    out = fn.Exp.apply(a)
    return out


def log(a: Tensor):
    a = atot(a)
    out = fn.Log.apply(a)
    return out


def sine(a: Tensor):
    a = atot(a)
    out = fn.Sine.apply(a)
    return out


def cosine(a: Tensor):
    a = atot(a)
    out = fn.Cosine.apply(a)
    return out


def sum(a: Tensor, dims: Optional[Union[Tuple[int, ...], int]] = None, keepdims=False):
    out = fn.Sum.apply(a, dims, keepdims)
    return out


def transpose(a: Tensor, dim_0=-2, dim_1=-1):
    out = fn.Tranpose.apply(a, dim_0, dim_1)
    return out


def permute(a: Tensor, dims: Optional[Tuple[int, ...]] = None):
    out = fn.Permute.apply(a, dims)
    return out


def squeeze(a: Tensor, dims: Optional[Union[Tuple[int, ...], int]] = None):
    if dims is None:
        a_dim = a.dim
        dims = tuple(i for i in range(len(a_dim)) if a_dim[i] == 1)
    out = fn.Squeeze.apply(a, dims=dims)
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
    a = atot(a)
    return fn.Abs.apply(a)


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
