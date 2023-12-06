import deepnet
import numpy as np


def zeros(dim, use_grad=False, dtype=None):
    zero_arr = np.zeros(dim)
    return deepnet.tensor(zero_arr, use_grad, dtype)


def zeros_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return deepnet.tensor(zero_arr, use_grad, dtype)


def ones(dim, use_grad=False, dtype=None):
    ones_arr = np.ones(dim)
    return deepnet.tensor(ones_arr, use_grad, dtype)


def ones_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data)
    return deepnet.tensor(ones_arr, use_grad, dtype)


def randn(dim=None, use_grad=False, dtype=None):
    if dim is None:
        dim = ()
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr, use_grad, dtype)


def rand(dim=None, use_grad=False, dtype=None):
    if dim is None:
        dim = ()
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr, use_grad, dtype)


def randint(low, high, dim, dtype=None):
    randint_arr = np.random.randint(low, high, dim, dtype)
    return deepnet.tensor(randint_arr)


def identity(n, use_grad=False, dtype=None):
    data = np.identity(n)
    return deepnet.tensor(data, use_grad, dtype)


def is_tensor(item):
    return isinstance(item, deepnet.Tensor)


def is_dual_tensor(item):
    return isinstance(item, deepnet.DualTensor)
