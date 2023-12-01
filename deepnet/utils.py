import deepnet
import numpy as np

# TODO these functions want a deepnet.dtype and not np.dtype


def zeros(dim, use_grad=False, dtype=None):
    zero_arr = np.zeros(dim, dtype)
    return deepnet.tensor(zero_arr, use_grad)


def zeros_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    zero_arr = np.zeros_like(data, dtype)
    return deepnet.tensor(zero_arr, use_grad)


def ones(dim, use_grad=False, dtype=None):
    ones_arr = np.ones(dim, dtype)
    return deepnet.tensor(ones_arr, use_grad)


def ones_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data, dtype)
    return deepnet.tensor(ones_arr, use_grad)


def randn(dim, use_grad=False):
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr, use_grad)


def rand(dim, use_grad=False):
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr, use_grad)


def randint(low, high, dim):
    randint_arr = np.random.randint(low, high, dim)
    return deepnet.tensor(randint_arr)
