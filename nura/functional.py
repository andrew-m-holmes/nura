import nura.functions as fn
from nura.tensors import Tensor, tensor
from nura.types import Tensorlike, Scalar, dimlike, dim
from typing import Optional, Union, Iterable


def add(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Add.apply(a, b)
    return out


def iadd(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a._data += b.data


def sub(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Sub.apply(a, b)
    return out


def isub(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a._data -= b.data


def mul(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Mul.apply(a, b)
    return out


def imul(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a._data *= b.data


def div(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Div.apply(a, b)
    return out


def idiv(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a._data /= b.data


def floordiv(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Floordiv.apply(a, b)
    return out


def ifloordiv(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a._data //= b.data


def modulo(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Modulo.apply(a, b)
    return out


def imodulo(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a._data %= b.data


def dot(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Dot.apply(a, b)
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("Cannot compute matmul() with tensors that aren't at least 2D")
    out = fn._Matmul.apply(a, b)
    return out


def imatmul(a: Tensor, b: Tensor) -> None:
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(
            "Cannot compute inplace matmul() with tensors that aren't at least 2D"
        )
    a._data @= b.data


def pow(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = fn._Pow.apply(a, b)
    return out


def ipow(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a._data **= b.data


def square(a: Tensor) -> Tensor:
    return fn._Pow.apply(a, tensor(2.0, dtype=a.dtype))


def sqrt(a: Tensor) -> Tensor:
    return fn._Pow.apply(a, tensor(0.5, dtype=a.dtype))


def exp(a: Tensor) -> Tensor:
    out = fn._Exp.apply(a)
    return out


def log(a: Tensor) -> Tensor:
    out = fn._Log.apply(a)
    return out


def sin(a: Tensor) -> Tensor:
    out = fn._Sin.apply(a)
    return out


def cos(a: Tensor) -> Tensor:
    out = fn._Cos.apply(a)
    return out


def sum(a: Tensor, dim: Optional[dimlike] = None, keepdims: bool = False) -> Tensor:
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Sum.apply(a, dim, keepdims)
    return out


def max(a: Tensor, dim: Optional[dimlike] = None, keepdims: bool = False) -> Tensor:
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Max.apply(a, dim, keepdims)
    return out


def min(a: Tensor, dim: Optional[dimlike] = None, keepdims: bool = False) -> Tensor:
    if dim is None:
        dim = tuple(range(a.ndim))
    out = fn._Min.apply(a, dim, keepdims)
    return out


def transpose(a: Tensor, dim0: int = -2, dim1: int = -1) -> Tensor:
    out = fn._Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor, dims: dim) -> Tensor:
    out = fn._Permute.apply(a, dims)
    return out


def squeeze(a: Tensor, dim: Optional[dimlike] = None) -> Tensor:
    out = fn._Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dim: dimlike) -> Tensor:
    out = fn._Unsqueeze.apply(a, dim)
    return out


def reshape(a: Tensor, newdim: dim) -> Tensor:
    out = fn._Reshape.apply(a, newdim)
    return out


def abs(a: Tensor) -> Tensor:
    return fn._Abs.apply(a)


def pos(a: Tensor) -> Tensor:
    return fn._Pos.apply(a)


def neg(a: Tensor) -> Tensor:
    return fn._Neg.apply(a)


def clone(a: Tensor) -> Tensor:
    out = fn._Clone.apply(a)
    return out


def select(a: Tensor, slice_: Union[Tensorlike, Tensor, slice]) -> Tensor:
    if isinstance(slice_, Iterable):
        slice_ = tuple(i.data if isinstance(i, Tensor) else i for i in slice_)
    if isinstance(slice_, Tensor):
        slice_ = slice_.data
    out = fn._Slice.apply(a, slice_)
    return out
