import numpy as np
from deepnet import Tensor
from deepnet.autograd.primitives import *


def add(a, b):
    out = Add.apply(a, b)
    return out


def sub(a, b):
    out = Sub.apply(a, b)
    return out


def mul(a, b):
    out = Mul.apply(a, b)
    return out


def div(a, b):
    out = Div.apply(a, b)
    return out


def matmul(a, b):
    out = Matmul.apply(a, b)
    return out


def pow(a, b):
    if not isinstance(b, Tensor):
        b = Tensor(b)
    out = Pow.apply(a, b)
    return out


def tranpose(tensor: Tensor, dim_0, dim_1):
    out = Tranpose.apply(tensor, dim_0, dim_1)
    return out
