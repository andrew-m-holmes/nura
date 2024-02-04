import numpy as np
import deepnet
from .tensors import Tensor


def zeros(dim, usegrad=False, dtype=None):
    dim = todim(dim)
    zero_arr = np.zeros(dim)
    return deepnet.tensor(zero_arr, usegrad, dtype)


def zeroslike(tensor: Tensor, usegrad=False, dtype=None):
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return deepnet.tensor(zero_arr, usegrad, dtype)


def ones(dim, usegrad=False, dtype=None):
    dim = todim(dim)
    ones_arr = np.ones(dim)
    return deepnet.tensor(ones_arr, usegrad, dtype)


def oneslike(tensor: Tensor, usegrad=False, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data)
    return deepnet.tensor(ones_arr, usegrad, dtype)


def randn(dim=None, usegrad=False, dtype=None):
    dim = todim(dim)
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr, usegrad, dtype)


def randnlike(tensor: Tensor, usegrad=False, dtype=None):
    dim = tensor.dim
    return randn(dim, usegrad, dtype)


def rand(dim=None, usegrad=False, dtype=None):
    dim = todim(dim)
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr, usegrad, dtype)


def randlike(tensor: Tensor, usegrad=False, dtype=None):
    dim = tensor.dim
    return rand(dim, usegrad, dtype)


def randint(low, high, dim, dtype=None):
    dim = todim(dim)
    randint_arr = np.random.randint(low, high, dim)
    return deepnet.tensor(randint_arr, dtype=dtype)


def randintlike(low, high, tensor: Tensor, dtype=None):
    dim = tensor.dim
    return randint(low, high, dim, dtype)


def identity(n, usegrad=False, dtype=None):
    data = np.identity(n)
    return deepnet.tensor(data, usegrad, dtype)


def full(dim, num, usegrad=False, dtype=None):
    dim = todim(dim)
    data = np.full(dim, num)
    return deepnet.tensor(data, usegrad, dtype)


def todim(dim):
    if dim is None:
        return tuple()
    if isinstance(dim, int):
        return (dim,)
    return dim


def iscontig(tensor):
    return tensor.data.flags["C_CONTIGUOUS"]


def istensor(obj):
    return isinstance(obj, Tensor)


def typename(tensor: Tensor):
    return tensor.__class__.__name__
