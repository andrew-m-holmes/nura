import numpy as np
import deepnet
from deepnet.types import dtype, _dim
from deepnet.tensors import Tensor, tensor
from typing import Optional, Type, Any, Tuple, Union


def zeros(
    dim: Union[_dim, int], usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = todim(dim)
    zero_arr = np.zeros(dim)
    return tensor(zero_arr, usegrad, dtype)


def zeroslike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    data = a.data
    zero_arr = np.zeros_like(data)
    return tensor(zero_arr, usegrad, dtype)


def ones(
    dim: Union[_dim, int], usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = todim(dim)
    ones_arr = np.ones(dim)
    return tensor(ones_arr, usegrad, dtype)


def oneslike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    data = a.data
    ones_arr = np.ones_like(data)
    return tensor(ones_arr, usegrad, dtype)


def randn(
    dim: Optional[Union[_dim, int]] = None,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    dim = todim(dim)
    randn_arr = np.random.randn(*dim)
    return tensor(randn_arr, usegrad, dtype)


def randnlike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    dim = a.dim
    return randn(dim, usegrad, dtype)


def rand(
    dim: Optional[Union[_dim, int]] = None,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    dim = todim(dim)
    rand_arr = np.random.rand(*dim)
    return tensor(rand_arr, usegrad, dtype)


def randlike(a: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    dim = a.dim
    return rand(dim, usegrad, dtype)


def randint(
    low: int, high: int, dim: Union[_dim, int], dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = todim(dim)
    randint_arr = np.random.randint(low, high, dim)
    return tensor(randint_arr, dtype=dtype)


def randintlike(
    low: int, high: int, a: Tensor, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = a.dim
    return randint(low, high, dim, dtype)


def identity(n: int, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    data = np.identity(n)
    return tensor(data, usegrad, dtype)


def full(
    dim: Union[_dim, int],
    num: float,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    dim = todim(dim)
    data = np.full(dim, num)
    return tensor(data, usegrad, dtype)


def eye(
    n: int,
    m: Optional[int] = None,
    k=0,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    data = np.eye(n, m, k)
    return tensor(data, dtype=dtype)


def any(a: Tensor, dim: Optional[Union[_dim, int]] = None, keepdims=False):
    return tensor(np.any(a.data, axis=dim, keepdims=keepdims))


def all(a: Tensor, dim: Optional[Union[_dim, int]] = None, keepdims=False):
    return tensor(np.all(a.data, axis=dim, keepdims=keepdims))


def hashtensor(a: Tensor) -> int:
    return hash(id(a))


def equal(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = deepnet.atot(a, b)
    return tensor(np.equal(a.data, b.data))


def less(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = atot(a, b)
    return tensor(np.less(a.data, b.data))


def lesseq(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = atot(a, b)
    return tensor(np.less_equal(a.data, b.data))


def greater(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = atot(a, b)
    return tensor(np.greater(a.data, b.data))


def greatereq(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = atot(a, b)
    return tensor(np.greater_equal(a.data, b.data))


def notequal(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = atot(a, b)
    return tensor(np.not_equal(a.data, b.data))


def tensorand(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = atot(a, b)
    return tensor(a.data and b.data)


def tensoror(a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
    a, b = atot(a, b)
    return tensor(a.data or b.data)


def tensornot(a: Union[Tensor, Any]) -> Tensor:
    b = atot(a)[0]
    return tensor(not b.data)


def atot(*args: Any) -> Union[Tuple[Tensor, ...], Tensor]:
    return tuple(a if istensor(a) else tensor(a) for a in args)


def to(a: Tensor, dtype: Type[dtype]):
    assert istensor(a)
    data = dtype.numpy(a.data)
    return tensor(data, a.usegrad, dtype)


def todim(dim: Any) -> Tuple[int, ...]:
    if dim is None:
        return tuple()
    if isinstance(dim, int):
        return (dim,)
    return dim


def iscontig(a: Tensor) -> bool:
    return a.data.flags["C_CONTIGUOUS"]


def istensor(obj: Any) -> bool:
    return isinstance(obj, Tensor)


def typename(a: Tensor) -> str:
    return a.__class__.__name__
