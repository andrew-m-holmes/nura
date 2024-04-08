import numpy as np
import nura.types as types
from nura.types import Scalar, dimlike, dim, dtype
from nura.tensors import Tensor, tensor
from typing import Optional, Type, Any, Tuple, Union


def empty(*dim: dimlike, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float
    dim = todim(dim)
    data = np.empty(dim, dtype=np.int8)
    return tensor(data, dtype=dtype)


def emptylike(a: Tensor, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float if dtype is types.bool else a.dtype
    data = a.data
    data = np.empty_like(data)
    return tensor(data, dtype=dtype)


def zeros(*dim: dimlike, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float
    dim = todim(dim)
    data = np.zeros(dim)
    return tensor(data, usegrad, dtype)


def zeroslike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float if dtype is types.bool else a.dtype
    data = a.data
    data = np.zeros_like(data)
    return tensor(data, usegrad, dtype)


def ones(*dim: dimlike, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float
    dim = todim(dim)
    data = np.ones(dim)
    return tensor(data, usegrad, dtype)


def oneslike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float if dtype is types.bool else a.dtype
    data = a.data
    data = np.ones_like(data)
    return tensor(data, usegrad, dtype)


def randn(
    *dim: dimlike,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    if dtype is None:
        dtype = types.float
    dim = todim(dim)
    data = np.random.randn(*dim)
    return tensor(data, usegrad, dtype)


def randnlike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = (
            types.float
            if a.dtype not in (types.half, types.float, types.double)
            else a.dtype
        )
    data = np.random.randn(*a.dim)
    return tensor(data, usegrad, dtype)


def rand(
    *dim: dimlike,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    if dtype is None:
        dtype = types.float
    dim = todim(dim)
    data = np.random.rand(*dim)
    return tensor(data, usegrad, dtype)


def randlike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = (
            types.float
            if a.dtype not in (types.half, types.float, types.double)
            else a.dtype
        )
    dim = a.dim
    return rand(dim, usegrad=usegrad, dtype=dtype)


def randint(
    *dim: dimlike, low: int, high: int, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    if dtype is None:
        dtype = types.int
    dim = todim(dim)
    data = np.random.randint(low, high, dim)
    return tensor(data, dtype=dtype)


def randintlike(
    low: int, high: int, a: Tensor, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    if dtype is None:
        dtype = (
            types.int
            if dtype in (types.half, types.float, types.double, types.bool)
            else a.dtype
        )
    data = np.random.randint(low, high, a.dim)
    return tensor(data, dtype=dtype)


def identity(n: int, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float
    data = np.identity(n)
    return tensor(data, dtype=dtype)


def tri(m: int, n: int, k=0, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = types.float
    data = np.tri(m, n, k)
    return tensor(data, dtype=dtype)


def triu(a: Tensor, k=0, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = a.dtype
    data = np.triu(a.data, k)
    return tensor(data, dtype=dtype)


def tril(a: Tensor, k=0, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = a.dtype
    data = np.tril(a.data, k)
    return tensor(data, dtype=dtype)


def full(
    *dim: dimlike,
    num: float,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    if dtype is None:
        dtype = types.float
    dim = todim(dim)
    data = np.full(dim, num)
    return tensor(data, usegrad, dtype)


def eye(
    n: int,
    m: Optional[int] = None,
    k=0,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    if dtype is None:
        dtype = types.float
    data = np.eye(n, m, k)
    return tensor(data, dtype=dtype)


def where(
    logical: Union[Tensor, bool],
    x: Union[Tensor, Scalar, bool],
    y: Union[Tensor, Scalar, bool],
) -> Tensor:
    data = logical.data if isinstance(logical, Tensor) else logical
    xdata = x.data if isinstance(x, Tensor) else x
    ydata = y.data if isinstance(y, Tensor) else y
    return tensor(np.where(data, xdata, ydata))


def indexwhere(logical: Union[Tensor, bool]) -> Tuple[Tensor, ...]:
    data = logical.data if isinstance(logical, Tensor) else logical
    return tuple(map(tensor, np.where(data)))


def nonzero(a: Tensor) -> Tuple[Tensor, ...]:
    arrs = np.nonzero(a.data)
    return tuple(map(tensor, arrs))


def argmax(a: Tensor, pos: Optional[int] = None, keepdims=False) -> Tensor:
    data = np.argmax(a.data, axis=pos, keepdims=keepdims)
    return tensor(data)


def argmin(a: Tensor, pos: Optional[int] = None, keepdims=False) -> Tensor:
    data = np.argmin(a.data, axis=pos, keepdims=keepdims)
    return tensor(data)


def hashtensor(a: Tensor) -> int:
    return hash(id(a))


def equal(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.equal(a.data, b.data))
    return tensor(np.equal(a.data, b))


def less(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.less(a.data, b.data))
    return tensor(np.less(a.data, b))


def lesseq(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.less_equal(a.data, b.data))
    return tensor(np.less_equal(a.data, b))


def greater(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.greater(a.data, b.data))
    return tensor(np.greater(a.data, b))


def greatereq(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.greater_equal(a.data, b.data))
    return tensor(np.greater_equal(a.data, b))


def notequal(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.not_equal(a.data, b.data))
    return tensor(np.not_equal(a.data, b))


def tensorany(a: Tensor, dim: Optional[dimlike] = None, keepdims=False) -> Tensor:
    return tensor(np.any(a.data, axis=dim, keepdims=keepdims))


def tensorall(a: Tensor, dim: Optional[dimlike] = None, keepdims=False) -> Tensor:
    return tensor(np.all(a.data, axis=dim, keepdims=keepdims))


def tensorand(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.logical_and(a.data, b.data))
    return tensor(np.logical_and(a.data, b))


def tensoror(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.logical_or(a.data, b.data))
    return tensor(np.logical_or(a.data, b))


def tensorxor(a: Tensor, b: Union[Tensor, Scalar, bool]) -> Tensor:
    if isinstance(b, Tensor):
        return tensor(np.logical_xor(a.data, b.data))
    return tensor(np.logical_xor(a.data, b))


def tensornot(a: Tensor) -> Tensor:
    return tensor(np.logical_not(a.data))


def tensorinvert(a: Tensor) -> Tensor:
    return tensor(np.invert(a.data))


def typesmatch(*tensors: Tensor) -> bool:
    return len(set(t.dtype for t in tensors)) == 1


def to(a: Tensor, dtype: Type[dtype]) -> Tensor:
    if not isinstance(a, Tensor):
        raise ValueError(f"Expected Tensor, received {a.__class__.__name__}")
    if a.usegrad and dtype not in (types.half, types.float, types.double):
        raise RuntimeError(
            f"Can't cast Tensor using gradient to type that doesn't, try Tensor.detach()"
        )
    data = dtype.numpy(a.data)
    return tensor(data, a.usegrad, dtype)


def tocontiguous(a: Tensor) -> Tensor:
    c = a.clone()
    data = np.ascontiguousarray(c.data)
    return c.mutated(data=data)


def todim(dim: Tuple[Any, ...]) -> dim:
    if not dim:
        return tuple()
    if isinstance(dim[0], tuple):
        return dim[0]
    return dim


def onehot(indices: Tensor, n: int, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if indices.ndim < 1 or indices.ndim > 2:
        raise ValueError(
            f"Expected indices with 1 or 2 dimensions, received {indices.ndim}"
        )
    if indices.dtype not in (types.int, types.long):
        raise TypeError(f"Expected int or long, received {indices.dtype.name()}")
    if dtype is None:
        dtype = indices.dtype
    return eye(n, dtype=dtype)[indices]


def iscontiguous(a: Tensor) -> bool:
    return a.data.flags["C_CONTIGUOUS"]


def typename(a: Tensor) -> str:
    return f"{a.dtype.name().capitalize()}{a.__class__.__name__}"
