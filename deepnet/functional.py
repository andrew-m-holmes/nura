import numpy as np
import deepnet
import deepnet.functions as funcs
from .autograd.mode import forward_ad_enabled
from .tensors import Tensor, DualTensor


def add(a, b):
    a, b = _preprocess_to_tensor(a, b)
    out = funcs.Add.apply(a, b)
    return out


def sub(a, b):
    a, b = _preprocess_to_tensor(a, b)
    out = funcs.Sub.apply(a, b)
    return out


def mul(a, b):
    a, b = _preprocess_to_tensor(a, b)
    out = funcs.Mul.apply(a, b)
    return out


def div(a, b):
    a, b = _preprocess_to_tensor(a, b)
    out = funcs.Div.apply(a, b)
    return out


def matmul(a, b):
    a, b = _preprocess_to_tensor(a, b)
    out = funcs.Matmul.apply(a, b)
    return out


def pow(a, b):
    a, b = _preprocess_to_tensor(a, b)
    out = funcs.Pow.apply(a, b)
    return out


def sine(a):
    a = _preprocess_to_tensor(a)
    out = funcs.Sine.apply(a)
    return out


def cosine(a):
    a = _preprocess_to_tensor(a)
    out = funcs.Cosine.apply(a)
    return out


def squeeze(a, dims=None):
    if dims is None:
        a_dim = a.dim()
        dims = tuple(i for i in range(len(a_dim)) if a_dim[i] == 1)
    out = funcs.Squeeze.apply(a, dims=dims)
    return out


def _preprocess_to_tensor(*args):
    tensor_fn = (
        deepnet.dual_tensor
        if forward_ad_enabled() else deepnet.tensor)
    assert _is_valid_tensor_data(*args)
    tensor_cls = [Tensor, DualTensor]
    tensors = (
        tensor_fn(arg)
        if type(arg) not in tensor_cls else arg for arg in args)
    return tensors


def _is_valid_tensor_data(*args):
    allowed_object_types = [
        Tensor, DualTensor, np.ndarray, int, float, bool]
    return all(type(arg) in allowed_object_types for arg in args)
