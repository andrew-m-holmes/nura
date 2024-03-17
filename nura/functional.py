import numpy as np
import nura.utils as utils
import nura.functions as fn
from nura.tensors import Tensor, tensor
from nura.types import Tensorlike, dimlike, dim
from typing import Optional


def add(a: Tensor | Tensorlike, b: Tensor | Tensorlike):
    a, b = utils.atot(a, b)
    out = fn._Add.apply(a, b)
    return out


def sub(a: Tensor | Tensorlike, b: Tensor | Tensorlike):
    a, b = utils.atot(a, b)
    out = fn._Sub.apply(a, b)
    return out


def mul(a: Tensor | Tensorlike, b: Tensor | Tensorlike):
    a, b = utils.atot(a, b)
    out = fn._Mul.apply(a, b)
    return out


def div(a: Tensor | Tensorlike, b: Tensor | Tensorlike):
    a, b = utils.atot(a, b)
    out = fn._Div.apply(a, b)
    return out


def dot(a: Tensor | Tensorlike, b: Tensor | Tensorlike):
    a, b = utils.atot(a, b)
    out = fn._Dot.apply(a, b)
    return out


def matmul(a: Tensor | Tensorlike, b: Tensor | Tensorlike):
    a, b = utils.atot(a, b)
    out = fn._Matmul.apply(a, b)
    return out


def pow(a: Tensor | Tensorlike, b: Tensor | Tensorlike):
    a, b = utils.atot(a, b)
    out = fn._Pow.apply(a, b)
    return out


def square(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    return fn._Pow.apply(a, tensor(2.0))


def sqrt(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    return fn._Pow.apply(a, tensor(0.5))


def exp(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    out = fn._Exp.apply(a)
    return out


def log(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    out = fn._Log.apply(a)
    return out


def sin(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    out = fn._Sin.apply(a)
    return out


def cos(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    out = fn._Cos.apply(a)
    return out


def sum(a: Tensor | Tensorlike, dim: Optional[dimlike] = None, keepdims=False):
    a = utils.atot(a)[0]
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Sum.apply(a, dim, keepdims)
    return out


def max(a: Tensor | Tensorlike, dim: Optional[dimlike] = None, keepdims=False):
    a = utils.atot(a)[0]
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Max.apply(a, dim, keepdims)
    return out


def min(a: Tensor | Tensorlike, dim: Optional[dimlike] = None, keepdims=False):
    a = utils.atot(a)[0]
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Min.apply(a, dim, keepdims)
    return out


def transpose(a: Tensor | Tensorlike, dim0=-2, dim1=-1):
    a = utils.atot(a)[0]
    out = fn._Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor | Tensorlike, dims: dim):
    a = utils.atot(a)[0]
    out = fn._Permute.apply(a, dims)
    return out


def squeeze(a: Tensor | Tensorlike, dim: Optional[dimlike] = None):
    a = utils.atot(a)[0]
    if dim is None:
        dim = tuple(np.where(np.array(a.dim) == 1)[0])
    out = fn._Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor | Tensorlike, dim: Optional[dimlike] = None):
    a = utils.atot(a)[0]
    if dim is None:
        dim = 0
    out = fn._Unsqueeze.apply(a, dim)
    return out


def view(a: Tensor | Tensorlike, newdim: dim):
    a = utils.atot(a)[0]
    out = fn._View.apply(a, newdim)
    return out


def reshape(a: Tensor | Tensorlike, newdim: dim):
    a = utils.atot(a)[0]
    a = tocontiguous(a)
    out = fn._Reshape.apply(a, newdim)
    return out


def abs(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    return fn._Abs.apply(a)


def pos(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    return fn._Pos.apply(a)


def neg(a: Tensor | Tensorlike):
    a = utils.atot(a)[0]
    return fn._Neg.apply(a)


def clone(a: Tensor):
    out = fn._Clone.apply(a)
    return out


def slice(a: Tensor, slc: slice):
    out = fn._Slice.apply(a, slc)
    return out


def tocontiguous(a: Tensor):
    cloned = a.clone()
    data = np.ascontiguousarray(cloned.data)
    return cloned.mutated(data=data)
