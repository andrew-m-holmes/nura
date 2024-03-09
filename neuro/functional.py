import numpy as np
import neuro.utils as utils
import neuro.functions as fn
from neuro.tensors import Tensor
from neuro.types import dim
from typing import Union, Optional, Any


def add(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn.Add.apply(a, b)
    return out


def sub(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn.Sub.apply(a, b)
    return out


def mul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn.Mul.apply(a, b)
    return out


def div(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn.Div.apply(a, b)
    return out


def dot(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    assert a.ndim >= 1 and b.ndim >= 1
    out = fn.Dot.apply(a, b)
    return out


def matmul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    assert a.ndim >= 2 and b.ndim >= 2
    assert a.dim[-1] == b.dim[-2]
    out = fn.Matmul.apply(a, b)
    return out


def pow(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn.Pow.apply(a, b)
    return out


def square(a: Union[Tensor, Any]):
    return pow(a, 2.0)


def sqrt(a: Union[Tensor, Any]):
    return pow(a, 0.5)


def exp(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    out = fn.Exp.apply(a)
    return out


def log(a: Union[Union[Tensor, Any], Any]):
    a = utils.atot(a)[0]
    out = fn.Log.apply(a)
    return out


def sin(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    out = fn.Sin.apply(a)
    return out


def cos(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    out = fn.Cos.apply(a)
    return out


def sum(a: Tensor, dim: Optional[Union[dim, int]] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn.Sum.apply(a, dim, keepdims)
    return out


def max(a: Tensor, dim: Optional[Union[dim, int]] = None, keepdims=False):
    b = utils.atot(a)[0]
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn.Max.apply(b, dim, keepdims)
    return out


def min(a: Tensor, dim: Optional[Union[dim, int]] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    a = utils.atot(a)[0]
    out = fn.Min.apply(a, dim, keepdims)
    return out


def transpose(a: Tensor, dim0=-2, dim1=-1):
    out = fn.Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor, dim: Optional[dim] = None):
    out = fn.Permute.apply(a, dim)
    return out


def squeeze(a: Tensor, dim: Optional[Union[dim, int]] = None):
    if dim is None:
        dim = tuple(np.where(np.array(a.dim) == 1)[0])
    out = fn.Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dim: Union[dim, int]):
    out = fn.Unsqueeze.apply(a, dim)
    return out


def view(a: Tensor, dim: dim):
    out = fn.View.apply(a, dim)
    return out


def reshape(a: Tensor, newdim: dim):
    a = tocontig(a)
    out = fn.Reshape.apply(a, newdim)
    return out


def abs(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    return fn.Abs.apply(a)


def pos(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    return fn.Pos.apply(a)


def neg(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    return fn.Neg.apply(a)


def clone(a: Tensor):
    out = fn.Clone.apply(a)
    return out


def slice(a: Tensor, slc: slice):
    out = fn.Slice.apply(a, slc)
    return out


def tocontig(a: Tensor):
    cloned = a.clone()
    data = np.ascontiguousarray(cloned.data)
    return cloned.mutated(data=data)
