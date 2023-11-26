import deepnet
import numpy as np


def zeros(dim, dtype=None):
    zero_arr = np.zeros(dim, dtype)
    return deepnet.tensor(zero_arr)


def zeros_like(tensor):
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return deepnet.tensor(zero_arr)


def ones(dim, dtype=None):
    ones_arr = np.ones(dim, dtype)
    return deepnet.tensor(ones_arr)


def ones_like(tensor, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data, dtype)
    return deepnet.tensor(ones_arr)


def randn(dim):
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr)


def rand(dim):
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr)


def randint(low, high, dim):
    randint_arr = np.random.randint(low, high, dim)
    return deepnet.tensor(randint_arr)
