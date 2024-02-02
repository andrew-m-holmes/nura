import numpy as np
import deepnet.utils as utils
import deepnet.functions as fn
from deepnet.tensors import Tensor

def add(a: Tensor, b: Tensor) -> Tensor:
    out = fn.Add.apply(a, b)
    return out


def sub(a, b):
    out = fn.Sub.apply(a, b)
    return out


def mul(a, b):
    out = fn.Mul.apply(a, b)
    return out


def div(a, b):
    out = fn.Div.apply(a, b)
    return out


def dot(a, b):
    out = fn.Dot.apply(a, b)
    return out


def matmul(a, b):
    if a.ndim() == 1:
        a = unsqueeze(a, 0)
    if b.ndim() == 1:
        b = unsqueeze(b, -1)
    out = fn.Matmul.apply(a, b)
    return out


def pow(a, b):
    out = fn.Pow.apply(a, b)
    return out


def exp(a):
    out = fn.Exp.apply(a)
    return out


def log(a):
    out = fn.Log.apply(a)
    return out


def sine(a):
    out = fn.Sine.apply(a)
    return out


def cosine(a):
    out = fn.Cosine.apply(a)
    return out


def sum(a, dims=None, keepdims=False):
    out = fn.Sum.apply(a, dims, keepdims)
    return out


def transpose(a, dim_0=-2, dim_1=-1):
    out = fn.Tranpose.apply(a, dim_0, dim_1)
    return out


def permute(a, dims=None):
    out = fn.Permute.apply(a, dims)
    return out


def squeeze(a, dims=None):
    if dims is None:
        a_dim = a.dim()
        dims = tuple(i for i in range(len(a_dim)) if a_dim[i] == 1)
    out = fn.Squeeze.apply(a, dims=dims)
    return out


def unsqueeze(a, dims):
    out = fn.Unsqueeze.apply(a, dims)
    return out


def view(a, dim):
    out = fn.View.apply(a, dim)
    return out


def reshape(a, dim):
    a = tocontig(a)
    out = fn.Reshape.apply(a, dim)
    return out


def clone(a):
    out = fn.Clone.apply(a)
    return out


def tocontig(tensor):
    if utils.iscontig(tensor):
        return tensor
    contiguous_tensor = tensor.clone()
    contiguous_tensor.data = np.ascontiguousarray(tensor.data)
    return contiguous_tensor


def slice(a, _slice):
    out = fn.Slice.apply(a, _slice)
    return out
