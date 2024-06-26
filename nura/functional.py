import nura.functions as functions
from nura.tensors import Tensor, tensor
from nura.types import Tensorlike, Scalar, dimlike, dim
from typing import Optional, Union, Iterable


def add(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = functions.Add.apply(a, b)
    return out


def iadd(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a.data += b.data


def sub(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = functions.Sub.apply(a, b)
    return out


def isub(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a.data -= b.data


def mul(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = functions.Mul.apply(a, b)
    return out


def imul(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a.data *= b.data


def div(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = functions.Div.apply(a, b)
    return out


def idiv(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a.data /= b.data


def floordiv(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = functions.Floordiv.apply(a, b)
    return out


def ifloordiv(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a.data //= b.data


def modulo(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = functions.Modulo.apply(a, b)
    return out


def imodulo(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a.data %= b.data


def dot(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Cannot compute dot product, tensors must 1D")
    out = functions.Dot.apply(a, b)
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError(
            "Cannot compute matrix multiplication, received one or more scalars, use mul()"
        )
    if a.ndim == 1 and b.ndim == 1:
        raise ValueError(
            "Cannot compute matrix multiplication, received vectors, use dot()"
        )
    out = functions.Matmul.apply(a, b)
    return out


def imatmul(a: Tensor, b: Tensor) -> None:
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError(
            "Cannot compute matrix multiplication, received one or more scalars, use mul()"
        )
    if a.ndim == 1 and b.ndim == 1:
        raise ValueError(
            "Cannot compute matrix multiplication, received vectors, use dot()"
        )
    a.data @= b.data


def pow(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    out = functions.Pow.apply(a, b)
    return out


def ipow(a: Tensor, b: Union[Tensor, Scalar]) -> None:
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=a.dtype)
    a.data **= b.data


def square(a: Tensor) -> Tensor:
    return functions.Pow.apply(a, tensor(2.0, dtype=a.dtype))


def sqrt(a: Tensor) -> Tensor:
    return functions.Pow.apply(a, tensor(0.5, dtype=a.dtype))


def exp(a: Tensor) -> Tensor:
    out = functions.Exp.apply(a)
    return out


def log(a: Tensor) -> Tensor:
    out = functions.Log.apply(a)
    return out


def sin(a: Tensor) -> Tensor:
    out = functions.Sin.apply(a)
    return out


def cos(a: Tensor) -> Tensor:
    out = functions.Cos.apply(a)
    return out


def sum(a: Tensor, dim: Optional[dimlike] = None, keepdims: bool = False) -> Tensor:
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Sum.apply(a, dim, keepdims)
    return out


def max(a: Tensor, dim: Optional[dimlike] = None, keepdims: bool = False) -> Tensor:
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Max.apply(a, dim, keepdims)
    return out


def min(a: Tensor, dim: Optional[dimlike] = None, keepdims: bool = False) -> Tensor:
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Min.apply(a, dim, keepdims)
    return out


def mean(a: Tensor, dim: Optional[dimlike] = None, keepdims: bool = False) -> Tensor:
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Mean.apply(a, dim, keepdims)
    return out


def var(
    a: Tensor,
    correction: int = 0,
    dim: Optional[dimlike] = None,
    keepdims: bool = False,
) -> Tensor:
    if correction < 0:
        raise ValueError("Cannot compute vairance with bias correct less than zero")
    if dim is None:
        dim = tuple(range(a.ndim))
    out = functions.Var.apply(a, correction, dim, keepdims)
    return out


def std(
    a: Tensor,
    correction: int = 0,
    dim: Optional[dimlike] = None,
    keepdims: bool = False,
) -> Tensor:
    return sqrt(var(a, correction, dim, keepdims))


def transpose(a: Tensor, dim0: int = -2, dim1: int = -1) -> Tensor:
    out = functions.Transpose.apply(a, dim0, dim1)
    return out


def permute(a: Tensor, dims: dim) -> Tensor:
    out = functions.Permute.apply(a, dims)
    return out


def squeeze(a: Tensor, dim: Optional[dimlike] = None) -> Tensor:
    if dim is None:
        dim = tuple(i for i, d in enumerate(a.dim) if d == 1)
    out = functions.Squeeze.apply(a, dim=dim)
    return out


def unsqueeze(a: Tensor, dim: dimlike) -> Tensor:
    out = functions.Unsqueeze.apply(a, dim)
    return out


def reshape(a: Tensor, newdim: dim) -> Tensor:
    out = functions.Reshape.apply(a, newdim)
    return out


def abs(a: Tensor) -> Tensor:
    return functions.Abs.apply(a)


def pos(a: Tensor) -> Tensor:
    return functions.Pos.apply(a)


def neg(a: Tensor) -> Tensor:
    return functions.Neg.apply(a)


def clone(a: Tensor) -> Tensor:
    out = functions.Clone.apply(a)
    return out


def select(
    a: Tensor,
    slice_: Union[
        Iterable[Union[Tensor, Tensorlike, slice]], Tensor, Tensorlike, slice
    ],
) -> Tensor:
    if isinstance(slice_, Iterable):
        slice_ = tuple(i.data if isinstance(i, Tensor) else i for i in slice_)
    if isinstance(slice_, Tensor):
        slice_ = slice_.data
    out = functions.Slice.apply(a, slice_)
    return out


def flatten(a: Tensor, start: int = 0, end: int = -1) -> Tensor:
    if a.ndim + end <= start:
        raise ValueError(
            "Cannot flatten Tensor, flatten ends at or before flatten starts"
        )
    return functions.Flatten.apply(a, start, end)


def concat(a: Tensor, b: Tensor, dim: int = 0) -> Tensor:
    if a.ndim != b.ndim:
        raise ValueError(
            "Cannot concatenate Tensors, they don't have the same number of dimensions"
        )
    dim = dim + a.ndim if dim < 0 else dim
    if a.dim[:dim] + a.dim[dim + 1 :] != b.dim[:dim] + b.dim[dim + 1 :]:
        raise ValueError(
            "Cannot concatenate Tensors, they differ for more than one dimension"
        )
    return functions.Concat.apply(a, b, dim)
