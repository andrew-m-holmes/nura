import numpy as np
import deepnet as dn
from deepnet.types import dtype, dim
from deepnet.tensors import Tensor
from typing import Optional, TypeGuard, Type, Any


def zeros(
    dim: dim, usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = todim(dim)
    zero_arr = np.zeros(dim)
    return dn.tensor(zero_arr, usegrad, dtype)


def zeroslike(
    tensor: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return dn.tensor(zero_arr, usegrad, dtype)


def ones(dim: dim, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    dim = todim(dim)
    ones_arr = np.ones(dim)
    return dn.tensor(ones_arr, usegrad, dtype)


def oneslike(
    tensor: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    data = tensor.data
    ones_arr = np.ones_like(data)
    return dn.tensor(ones_arr, usegrad, dtype)


def randn(
    dim: Optional[dim] = None,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    dim = todim(dim)
    randn_arr = np.random.randn(*dim)
    return dn.tensor(randn_arr, usegrad, dtype)


def randnlike(
    tensor: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = tensor.dim
    return randn(dim, usegrad, dtype)


def rand(
    dim: Optional[dim] = None,
    usegrad=False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    dim = todim(dim)
    rand_arr = np.random.rand(*dim)
    return dn.tensor(rand_arr, usegrad, dtype)


def randlike(
    tensor: Tensor, usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = tensor.dim
    return rand(dim, usegrad, dtype)


def randint(low: int, high: int, dim: dim, dtype: Optional[Type[dtype]] = None) -> Tensor:
    dim = todim(dim)
    randint_arr = np.random.randint(low, high, dim)
    return dn.tensor(randint_arr, dtype=dtype)


def randintlike(
    low: int, high: int, tensor: Tensor, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    dim = tensor.dim
    return randint(low, high, dim, dtype)


def identity(n: int, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    data = np.identity(n)
    return dn.tensor(data, usegrad, dtype)


def full(dim: dim, num: float, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    dim = todim(dim)
    data = np.full(dim, num)
    return dn.tensor(data, usegrad, dtype)


def to(tensor: Tensor, dtype: Type[dtype]):
    assert istensor(tensor)
    data = dtype.numpy(tensor.data)
    return dn.tensor(data, tensor.usegrad, dtype)


def todim(dim: Any):
    if dim is None:
        return tuple()
    if isinstance(dim, int):
        return (dim,)
    return dim


def iscontig(tensor: Tensor) -> bool:
    return tensor.data.flags["C_CONTIGUOUS"]


def istensor(obj: Any) -> TypeGuard[Tensor]:
    return isinstance(obj, Tensor)


def typename(tensor: Tensor) -> str:
    return tensor.__class__.__name__
