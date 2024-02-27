import numpy as np
import deepnet.utils as utils
import deepnet.functions as functions
from deepnet.tensors import Tensor
from deepnet.types import _dim
from typing import Union, Optional, Any


def add(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = functions.Add.apply(a, b)
    return out


def sub(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = functions.Sub.apply(a, b)
    return out


def mul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = functions.Mul.apply(a, b)
    return out


def div(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = functions.Div.apply(a, b)
    return out


def dot(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    assert a.ndim >= 1 and b.ndim >= 1
    out = functions.Dot.apply(a, b)
    return out


def matmul(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    assert a.ndim >= 2 and b.ndim >= 2
    assert a.dim[-1] == b.dim[-2]
    out = functions.Matmul.apply(a, b)
    return out


def pow(a: Union[Tensor, Any], b: Union[Tensor, Any]):
    a, b = utils.atot(a, b)
    out = functions.Pow.apply(a, b)
    return out


def square(a: Union[Tensor, Any]):
    return pow(a, 2.0)


def sqrt(a: Union[Tensor, Any]):
    return pow(a, 0.5)


def exp(a: Union[Tensor, Any]):
    b = utils.atot(a)[0]
    out = functions.Exp.apply(b)
    return out


def log(a: Union[Union[Tensor, Any], Any]):
    b = utils.atot(a)[0]
    out = functions.Log.apply(b)
    return out


def sin(a: Union[Tensor, Any]):
    b = utils.atot(a)[0]
    out = functions.Sin.apply(b)
    return out


def cos(a: Union[Tensor, Any]):
    b = utils.atot(a)[0]
    out = functions.Cos.apply(b)
    return out


def sum(a: Tensor, dim: Optional[Union[_dim, int]] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Sum.apply(a, dim, keepdims)
    return out


def max(a: Tensor, dim: Optional[Union[_dim, int]] = None, keepdims=False):
    b = utils.atot(a)[0]
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Max.apply(b, dim, keepdims)
    return out


def min(a: Tensor, dim: Optional[Union[_dim, int]] = None, keepdims=False):
    b = utils.atot(a)[0]
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Min.apply(b, dim, keepdims)
    return out


def transpose(a: Tensor, dim0=-2, dim1=-1):
    out = functions.Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor, dim: Optional[_dim] = None):
    out = functions.Permute.apply(a, dim)
    return out


def squeeze(a: Tensor, dim: Optional[Union[_dim, int]] = None):
    if dim is None:
        dim = tuple(np.where(np.array(a.dim) == 1)[0])
    out = functions.Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dim: Union[_dim, int]):
    out = functions.Unsqueeze.apply(a, dim)
    return out


def view(a: Tensor, dim: _dim):
    out = functions.View.apply(a, dim)
    return out


def reshape(a: Tensor, newdim: _dim):
    a = tocontig(a)
    out = functions.Reshape.apply(a, newdim)
    return out


def abs(a: Union[Tensor, Any]):
    b = utils.atot(a)[0]
    return functions.Abs.apply(b)


def clone(a: Tensor):
    out = functions.Clone.apply(a)
    return out


def slice(a: Tensor, slc: slice):
    out = functions.Slice.apply(a, slc)
    return out


def tocontig(a: Tensor):
    b = a.clone()
    data = np.ascontiguousarray(b.data)
    return b.mutated(data=data)
