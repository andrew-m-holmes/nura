from .primitives import *


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
    out = Pow.apply(a, b)
    return out


def tranpose(a, dim_0, dim_1):
    out = Tranpose.apply(a, dim_0, dim_1)
    return out


def clone(a):
    out = Clone.apply(a)
    return out
