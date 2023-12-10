import deepnet.functions as funcs
from .utils import preprocess_to_tensor


def add(a, b):
    a, b = preprocess_to_tensor(a, b)
    out = funcs.Add.apply(a, b)
    return out


def sub(a, b):
    a, b = preprocess_to_tensor(a, b)
    out = funcs.Sub.apply(a, b)
    return out


def mul(a, b):
    a, b = preprocess_to_tensor(a, b)
    out = funcs.Mul.apply(a, b)
    return out


def div(a, b):
    a, b = preprocess_to_tensor(a, b)
    out = funcs.Div.apply(a, b)
    return out


def matmul(a, b):
    a, b = preprocess_to_tensor(a, b)
    out = funcs.Matmul.apply(a, b)
    return out


def pow(a, b):
    a, b = preprocess_to_tensor(a, b)
    out = funcs.Pow.apply(a, b)
    return out


def sine(a):
    a = preprocess_to_tensor(a)
    out = funcs.Sine.apply(a)
    return out


def cosine(a):
    a = preprocess_to_tensor(a)
    out = funcs.Cosine.apply(a)
    return out


def sum(a, dims=None, keepdims=True):
    a = preprocess_to_tensor(a)
    out = funcs.Sum.apply(a, dims, keepdims)
    return out


def squeeze(a, dims=None):
    if dims is None:
        a_dim = a.dim()
        dims = tuple(i for i in range(len(a_dim)) if a_dim[i] == 1)
    out = funcs.Squeeze.apply(a, dims=dims)
    return out
