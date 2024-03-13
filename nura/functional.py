import numpy as np
import nura.utils as utils
import nura.functions as fn
from nura.tensors import Tensor
from nura.types import dim, dimlike
from typing import Union, Optional, Any


def add(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn._Add.apply(a, b)
    return out


def sub(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn._Sub.apply(a, b)
    return out


def mul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn._Mul.apply(a, b)
    return out


def div(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn._Div.apply(a, b)
    return out


def dot(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    assert a.ndim >= 1 and b.ndim >= 1
    assert utils.typesmatch(a, b)
    out = fn._Dot.apply(a, b)
    return out


def matmul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    assert a.ndim >= 2 and b.ndim >= 2
    assert a.dim[-1] == b.dim[-2]
    assert utils.typesmatch(a, b)
    out = fn._Matmul.apply(a, b)
    return out


def pow(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = fn._Pow.apply(a, b)
    return out


def square(a: Union[Tensor, Any]):
    return pow(a, 2.0)


def sqrt(a: Union[Tensor, Any]):
    return pow(a, 0.5)


def exp(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    out = fn._Exp.apply(a)
    return out


def log(a: Union[Union[Tensor, Any], Any]):
    a = utils.atot(a)[0]
    out = fn._Log.apply(a)
    return out


def sin(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    out = fn._Sin.apply(a)
    return out


def cos(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    out = fn._Cos.apply(a)
    return out


def sum(a: Tensor, dim: Optional[dimlike] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Sum.apply(a, dim, keepdims)
    return out


def max(a: Tensor, dim: Optional[dimlike] = None, keepdims=False):
    b = utils.atot(a)[0]
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Max.apply(b, dim, keepdims)
    return out


def min(a: Tensor, dim: Optional[dimlike] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    a = utils.atot(a)[0]
    out = fn._Min.apply(a, dim, keepdims)
    return out


def transpose(a: Tensor, dim0=-2, dim1=-1):
    out = fn._Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor, dims: Optional[dim] = None):
    out = fn._Permute.apply(a, dims)
    return out


def squeeze(a: Tensor, dim: Optional[dimlike] = None):
    if dim is None:
        dim = tuple(np.where(np.array(a.dim) == 1)[0])
    out = fn._Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dim: Optional[dimlike] = None):
    if dim is None:
        dim = 0
    out = fn._Unsqueeze.apply(a, dim)
    return out


def view(a: Tensor, newdim: dim):
    out = fn._View.apply(a, newdim)
    return out


def reshape(a: Tensor, newdim: dim):
    a = tocontig(a)
    out = fn._Reshape.apply(a, newdim)
    return out


def abs(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    return fn._Abs.apply(a)


def pos(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    return fn._Pos.apply(a)


def neg(a: Union[Tensor, Any]):
    a = utils.atot(a)[0]
    return fn._Neg.apply(a)


def clone(a: Tensor):
    out = fn._Clone.apply(a)
    return out


def slice(a: Tensor, slc: slice):
    out = fn._Slice.apply(a, slc)
    return out


def tocontig(a: Tensor):
    cloned = a.clone()
    data = np.ascontiguousarray(cloned.data)
    return cloned.mutated(data=data)
