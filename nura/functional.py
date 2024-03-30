import nura.utils as utils
import nura.functions as fn
from nura.tensors import Tensor, tensor
from nura.types import Scalar, dimlike, dim
from typing import Optional, Union, Any


def add(a: Tensor, b: Union[Tensor, Scalar]):
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Add.apply(a, b)
    return out


def sub(a: Tensor, b: Union[Tensor, Scalar]):
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Sub.apply(a, b)
    return out


def mul(a: Tensor, b: Union[Tensor, Scalar]):
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Mul.apply(a, b)
    return out


def div(a: Tensor, b: Union[Tensor, Scalar]):
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Div.apply(a, b)
    return out


def dot(a: Tensor, b: Tensor):
    out = fn._Dot.apply(a, b)
    return out


def matmul(a: Tensor, b: Tensor):
    out = fn._Matmul.apply(a, b)
    return out


def pow(a: Tensor, b: Union[Tensor, Scalar]):
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Pow.apply(a, b)
    return out


def square(a: Tensor):
    return fn._Pow.apply(a, tensor(2.0))


def sqrt(a: Tensor):
    return fn._Pow.apply(a, tensor(0.5))


def exp(a: Tensor):
    out = fn._Exp.apply(a)
    return out


def log(a: Tensor):
    out = fn._Log.apply(a)
    return out


def sin(a: Tensor):
    out = fn._Sin.apply(a)
    return out


def cos(a: Tensor):
    out = fn._Cos.apply(a)
    return out


def sum(a: Tensor, dim: Optional[dimlike] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Sum.apply(a, dim, keepdims)
    return out


def max(a: Tensor, dim: Optional[dimlike] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Max.apply(a, dim, keepdims)
    return out


def min(a: Tensor, dim: Optional[dimlike] = None, keepdims=False):
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Min.apply(a, dim, keepdims)
    return out


def transpose(a: Tensor, dim0=-2, dim1=-1):
    out = fn._Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor, dims: dim):
    out = fn._Permute.apply(a, dims)
    return out


def squeeze(a: Tensor, dim: Optional[dimlike] = None):
    if dim is None:
        dim = tuple(i for i, d in enumerate(a.dim) if d == 1)
    out = fn._Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dim: dimlike):
    out = fn._Unsqueeze.apply(a, dim)
    return out


def view(a: Tensor, newdim: dim):
    out = fn._View.apply(a, newdim)
    return out


def reshape(a: Tensor, newdim: dim):
    a = utils.tocontiguous(a)
    out = fn._Reshape.apply(a, newdim)
    return out


def abs(a: Tensor):
    return fn._Abs.apply(a)


def pos(a: Tensor):
    return fn._Pos.apply(a)


def neg(a: Tensor):
    return fn._Neg.apply(a)


def clone(a: Tensor):
    out = fn._Clone.apply(a)
    return out


def slice(a: Tensor, slc: Any):
    if isinstance(slc, tuple):
        slc = tuple(i.data if isinstance(i, Tensor) else i for i in slc)
    if isinstance(slc, Tensor):
        slc = slc.int().data
    out = fn._Slice.apply(a, slc)
    return out
