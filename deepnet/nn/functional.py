import deepnet
from deepnet.autograd.primitives import *


def add(a, b):
    a, b = _format_to_tensor(a, b)
    out = Add.apply(a, b)
    return out


def sub(a, b):
    a, b = _format_to_tensor(a, b)
    out = Sub.apply(a, b)
    return out


def mul(a, b):
    a, b = _format_to_tensor(a, b)
    out = Mul.apply(a, b)
    return out


def div(a, b):
    a, b = _format_to_tensor(a, b)
    out = Div.apply(a, b)
    return out


def matmul(a, b):
    a, b = _format_to_tensor(a, b)
    out = Matmul.apply(a, b)
    return out


def pow(a, b):
    a, b = _format_to_tensor(a, b)
    out = Pow.apply(a, b)
    return out


def tranpose(a, dim_0, dim_1):
    a = _format_to_tensor(a)[0]
    out = Tranpose.apply(a, dim_0, dim_1)
    return out


def clone(a):
    a = _format_to_tensor(a)[0]
    out = Clone.apply(a)
    return out


def _format_to_tensor(*inputs):
    tensors = []
    for input in inputs:
        if not isinstance(input, Tensor):
            tensors.append(deepnet.tensor(input))
        else:
            tensors.append(input)
    return tensors
