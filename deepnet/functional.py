import numpy as np
from . import functions
from . import utils


def add(a, b):
    out = functions.Add.apply(a, b)
    return out


def sub(a, b):
    out = functions.Sub.apply(a, b)
    return out


def mul(a, b):
    out = functions.Mul.apply(a, b)
    return out


def div(a, b):
    out = functions.Div.apply(a, b)
    return out


def dot(a, b):
    out = functions.Dot.apply(a, b)
    return out


def matmul(a, b):
    if a.ndim() == 1:
        a = unsqueeze(a, 0)
    if b.ndim() == 1:
        b = unsqueeze(b, -1)
    out = functions.Matmul.apply(a, b)
    return out


def pow(a, b):
    out = functions.Pow.apply(a, b)
    return out


def exp(a):
    out = functions.Exp.apply(a)
    return out


def log(a):
    out = functions.Log.apply(a)
    return out


def sine(a):
    out = functions.Sine.apply(a)
    return out


def cosine(a):
    out = functions.Cosine.apply(a)
    return out


def sum(a, dims=None, keepdims=False):
    out = functions.Sum.apply(a, dims, keepdims)
    return out


def transpose(a, dim_0=-2, dim_1=-1):
    out = functions.Tranpose.apply(a, dim_0, dim_1)
    return out


def permute(a, dims=None):
    out = functions.Permute.apply(a, dims)
    return out


def squeeze(a, dims=None):
    if dims is None:
        a_dim = a.dim()
        dims = tuple(i for i in range(len(a_dim)) if a_dim[i] == 1)
    out = functions.Squeeze.apply(a, dims=dims)
    return out


def unsqueeze(a, dims):
    out = functions.Unsqueeze.apply(a, dims)
    return out


def view(a, dim):
    out = functions.View.apply(a, dim)
    return out


def reshape(a, dim):
    a = to_contiguous(a)
    out = functions.Reshape.apply(a, dim)
    return out


def clone(a):
    out = functions.Clone.apply(a)
    return out


def to_contiguous(tensor):
    if utils.is_contiguous(tensor):
        return tensor
    contiguous_tensor = tensor.clone()
    contiguous_tensor.data = np.ascontiguousarray(tensor.data)
    return contiguous_tensor


def slice(a, _slice):
    out = functions.Slice.apply(a, _slice)
    return out
