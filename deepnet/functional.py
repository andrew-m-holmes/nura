import numpy as np
import deepnet.utils as utils
import deepnet.functions as fn
from deepnet.tensors import Tensor, tensor
from deepnet.types import _dim
from typing import Union, Tuple, Optional, Any


def add(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = atot(a, b)
    out = fn.Add.apply(a, b)
    return out


def sub(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = atot(a, b)
    out = fn.Sub.apply(a, b)
    return out


def mul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = atot(a, b)
    out = fn.Mul.apply(a, b)
    return out


def div(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = atot(a, b)
    out = fn.Div.apply(a, b)
    return out


def dot(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = atot(a, b)
    out = fn.Dot.apply(a, b)
    return out


def matmul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = atot(a, b)
    assert a.ndim >= 1 and b.ndim >= 1
    if a.ndim == 1:
        a = unsqueeze(a, 0)
    if b.ndim == 1:
        b = unsqueeze(b, -1)
    out = fn.Matmul.apply(a, b)
    return out


def pow(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = atot(a, b)
    out = fn.Pow.apply(a, b)
    return out


def exp(a: Union[Tensor, Any]):
    b = atot(a)[0]
    out = fn.Exp.apply(b)
    return out


def log(a: Union[Union[Tensor, Any], Any]):
    b = atot(a)[0]
    out = fn.Log.apply(b)
    return out


def sin(a: Union[Tensor, Any]):
    b = atot(a)[0]
    out = fn.Sin.apply(b)
    return out


def cos(a: Union[Tensor, Any]):
    b = atot(a)[0]
    out = fn.Cos.apply(b)
    return out


def sum(a: Tensor, dim: Optional[Union[_dim, int]] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn.Sum.apply(a, dim, keepdims)
    return out


def transpose(a: Tensor, dim0=-2, dim1=-1):
    out = fn.Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor, dim: Optional[_dim] = None):
    out = fn.Permute.apply(a, dim)
    return out


def squeeze(a: Tensor, dim: Optional[Union[_dim, int]] = None):
    if dim is None:
        dim = tuple(np.where(np.array(a.dim) == 1)[0])
    out = fn.Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dim: Union[_dim, int]):
    out = fn.Unsqueeze.apply(a, dim)
    return out


def view(a: Tensor, dim: _dim):
    out = fn.View.apply(a, dim)
    return out


def reshape(a: Tensor, newdim: _dim):
    a = tocontig(a)
    out = fn.Reshape.apply(a, newdim)
    return out


def abs(a: Union[Tensor, Any]):
    b = atot(a)[0]
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


def slice(a: Tensor, slc: slice):
    out = fn.Slice.apply(a, slc)
    return out


def atot(*args: Any) -> Union[Tuple[Tensor, ...], Tensor]:
    return tuple(a if utils.istensor(a) else tensor(a) for a in args)
