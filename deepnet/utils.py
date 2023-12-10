import numpy as np
import deepnet
from deepnet import Tensor, DualTensor


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


def full(dim, num, use_grad=False, dtype=None):
    data = np.full(dim, num)
    return deepnet.tensor(data, use_grad, dtype)


def is_tensor(item):
    return isinstance(item, Tensor)


def is_dual_tensor(item):
    return isinstance(item, DualTensor)


def preprocess_to_tensor(*args):
    tensor_fn = (
        deepnet.dual_tensor
        if deepnet.forward_ad_enabled() else deepnet.tensor)
    assert is_valid_tensor_data(*args)
    tensor_cls = [Tensor, DualTensor]
    tensors = tuple(
        tensor_fn(arg)
        if type(arg) not in tensor_cls else arg for arg in args)
    return tensors if len(tensors) > 1 else tensors[0]


def is_valid_tensor_data(*args):
    allowed_object_types = [
        Tensor, DualTensor, np.ndarray, int, float, bool]
    return all(type(arg) in allowed_object_types for arg in args)
