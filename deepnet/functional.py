import numpy as np
import deepnet.utils as utils
import deepnet.functions as funcs


def add(a, b):
    a, b = utils.preprocess_to_tensors(a, b)
    out = funcs.Add.apply(a, b)
    return out


def sub(a, b):
    a, b = utils.preprocess_to_tensors(a, b)
    out = funcs.Sub.apply(a, b)
    return out


def mul(a, b):
    a, b = utils.preprocess_to_tensors(a, b)
    out = funcs.Mul.apply(a, b)
    return out


def div(a, b):
    a, b = utils.preprocess_to_tensors(a, b)
    out = funcs.Div.apply(a, b)
    return out


def dot(a, b):
    _dot_args_check(a, b)
    out = funcs.Dot.apply(a, b)
    return out


def _dot_args_check(a, b):
    assert utils.is_all_tensor(a, b)
    assert a.dim() == b.dim()
    assert a.ndim() == b.ndim() == 1


def matmul(a, b):
    _matmul_args_check(a, b)
    a, b = _setup_tensors_for_matmul(a, b)
    out = funcs.Matmul.apply(a, b)
    return out


def _setup_tensors_for_matmul(a, b):
    if utils.is_vector_tensor(a):
        a = unsqueeze(a, 0)
    if utils.is_vector_tensor(b):
        b = unsqueeze(b, -1)
    return a, b

def _matmul_args_check(a, b):
    assert utils.is_all_tensor(a, b)
    assert a.ndim() >= 1
    assert b.ndim() >= 1

def pow(a, b):
    a, b = utils.preprocess_to_tensors(a, b)
    out = funcs.Pow.apply(a, b)
    return out


def exp(a):
    a = utils.preprocess_to_tensors(a)
    out = funcs.Exp.apply(a)
    return out


def log(a):
    a = utils.preprocess_to_tensors(a)
    out = funcs.Log.apply(a)
    return out


def sine(a):
    a = utils.preprocess_to_tensors(a)
    out = funcs.Sine.apply(a)
    return out


def cosine(a):
    a = utils.preprocess_to_tensors(a)
    out = funcs.Cosine.apply(a)
    return out


def sum(a, dims=None, keepdims=False):
    _sum_args_check(a, dims, keepdims)
    out = funcs.Sum.apply(a, dims, keepdims)
    return out


def _sum_args_check(a, dims, keepdims):
    assert utils.is_tensor(a)
    assert utils.is_dims_arg(dims)
    assert utils.is_py_bool(keepdims)


def transpose(a, dim_0=-2, dim_1=-1):
    _transpose_args_check(a, dim_0, dim_1)
    out = funcs.Tranpose.apply(a, dim_0, dim_1)
    return out


def _transpose_args_check(a, dim_0, dim_1):
    assert utils.is_tensor(a)
    assert a.ndim() >= 2
    assert utils.is_all_py_scalars(dim_0, dim_1)


def permute(a, dims=None):
    _permute_args_check(a, dims)
    out = funcs.Permute.apply(a, dims)
    return out


def _permute_args_check(a, dims):
    assert utils.is_tensor(a)
    assert utils.is_dims_arg(dims)


def squeeze(a, dims=None):
    _squeeze_args_check(a, dims)
    dims = _setup_dims_for_squeeze(a, dims)
    out = funcs.Squeeze.apply(a, dims=dims)
    return out


def _setup_dims_for_squeeze(a, dims):
    if dims is None:
        a_dim = a.dim()
        dims = tuple(i for i in range(len(a_dim)) if a_dim[i] == 1)
    return dims


def _squeeze_args_check(a, dims):
    assert utils.is_tensor(a)
    assert utils.is_dims_arg(dims)


def unsqueeze(a, dims):
    _squeeze_args_check(a, dims)
    out = funcs.Unsqueeze.apply(a, dims)
    return out


def view(a, dim):
    _view_args_check(a, dim)
    out = funcs.View.apply(a, dim)
    return out


def _view_args_check(a, dim):
    assert utils.is_tensor(a)
    assert utils.is_contiguous(a)
    assert utils.is_dims_arg(dim)


def reshape(a, dim):
    _reshape_args_check(a, dim)
    a = to_contiguous(a)
    out = funcs.Reshape.apply(a, dim)
    return out


def _reshape_args_check(a, dim):
    assert utils.is_tensor(a)
    assert utils.is_dims_arg(dim)


def clone(a):
    assert utils.is_tensor(a)
    out = funcs.Clone.apply(a)
    return out

def to_contiguous(tensor):
    if utils.is_contiguous(tensor):
        return tensor
    contiguous_tensor = tensor.clone()
    contiguous_tensor.data = np.ascontiguousarray(tensor.data)
    return contiguous_tensor


def slice(a, _slice):
    assert utils.is_tensor(a)
    out = funcs.Slice.apply(a, _slice)
    return out
