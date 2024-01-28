import numpy as np

import deepnet
from deepnet import Tensor
from deepnet.dtype import dtype


def zeros(dim, diff=False, dtype=None):
    dim = todim(dim)
    zero_arr = np.zeros(dim)
    return deepnet.tensor(zero_arr, diff, dtype)


def zeros_like(tensor, diff=False, dtype=None):
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return deepnet.tensor(zero_arr, diff, dtype)


def ones(dim, diff=False, dtype=None):
    dim = todim(dim)
    ones_arr = np.ones(dim)
    return deepnet.tensor(ones_arr, diff, dtype)


def ones_like(tensor, diff=False, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data)
    return deepnet.tensor(ones_arr, diff, dtype)


def randn(dim=None, diff=False, dtype=None):
    dim = todim(dim)
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr, diff, dtype)


def rand(dim=None, diff=False, dtype=None):
    dim = todim(dim)
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr, diff, dtype)


def randint(low, high, dim, dtype=None):
    dim = todim(dim)
    randint_arr = np.random.randint(low, high, dim)
    return deepnet.tensor(randint_arr, dtype=dtype)


def identity(n, diff=False, dtype=None):
    data = np.identity(n)
    return deepnet.tensor(data, diff, dtype)


def full(dim, num, diff=False, dtype=None):
    dim = todim(dim)
    data = np.full(dim, num)
    return deepnet.tensor(data, diff, dtype)


def todim(dim):
    if dim is None:
        return tuple()
    if isinstance(dim, int):
        return (dim,)
    return dim

def is_contiguous(tensor):
    return tensor.data.flags["C_CONTIGUOUS"]
