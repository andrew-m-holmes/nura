import numpy as np
import deepnet
from .tensors import Tensor


def zeros(dim, mut=False, dtype=None):
    dim = todim(dim)
    zero_arr = np.zeros(dim)
    return deepnet.tensor(zero_arr, mut, dtype)


def zeroslike(tensor: Tensor, mut=False, dtype=None):
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return deepnet.tensor(zero_arr, mut, dtype)


def ones(dim, mut=False, dtype=None):
    dim = todim(dim)
    ones_arr = np.ones(dim)
    return deepnet.tensor(ones_arr, mut, dtype)


def oneslike(tensor: Tensor, mut=False, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data)
    return deepnet.tensor(ones_arr, mut, dtype)


def randn(dim=None, mut=False, dtype=None):
    dim = todim(dim)
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr, mut, dtype)


def rand(dim=None, mut=False, dtype=None):
    dim = todim(dim)
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr, mut, dtype)


def randint(low, high, dim, dtype=None):
    dim = todim(dim)
    randint_arr = np.random.randint(low, high, dim)
    return deepnet.tensor(randint_arr, dtype=dtype)


def identity(n, mut=False, dtype=None):
    data = np.identity(n)
    return deepnet.tensor(data, mut, dtype)


def full(dim, num, mut=False, dtype=None):
    dim = todim(dim)
    data = np.full(dim, num)
    return deepnet.tensor(data, mut, dtype)


def todim(dim):
    if dim is None:
        return tuple()
    if isinstance(dim, int):
        return (dim,)
    return dim

def iscontig(tensor):
    return tensor.data.flags["C_CONTIGUOUS"]

def istensor(obj):
    return isinstance(obj, deepnet.Tensor)
