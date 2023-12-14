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


def matmul(a, b):
    assert utils.is_of_tensor(a, b)
    out = funcs.Matmul.apply(a, b)
    return out


def pow(a, b):
    a, b = utils.preprocess_to_tensors(a, b)
    out = funcs.Pow.apply(a, b)
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
    assert _valid_sum_args(a, dims, keepdims)
    out = funcs.Sum.apply(a, dims, keepdims)
    return out


def _valid_sum_args(a, dims, keepdims):
    passed = utils.is_of_tensor(a) and utils.is_dims_arg(
        dims) and utils.is_py_bool(keepdims)
    return passed


def transpose(a, dim_0=-2, dim_1=-1):
    assert _valid_transpose_args(a, dim_0, dim_1)
    out = funcs.Tranpose.apply(a, dim_0, dim_1)
    return out


def _valid_transpose_args(a, dim_0, dim_1):
    return utils.is_of_tensor(a) and utils.is_all_py_scalars(
        dim_0, dim_1) and a.ndim() >= 2


def permute(a, dims=None):
    assert _valid_permute_args(a, dims)
    out = funcs.Permute.apply(a, dims)
    return out


def _valid_permute_args(a, dims):
    return utils.is_tensor(a) and utils.is_dims_arg(dims)


def squeeze(a, dims=None):
    assert _valid_squeeze_args(a, dims)
    dims = _setup_dims_for_squeeze(a, dims)
    out = funcs.Squeeze.apply(a, dims=dims)
    return out


def _setup_dims_for_squeeze(a, dims):
    if dims is None:
        a_dim = a.dim()
        dims = tuple(i for i in range(len(a_dim)) if a_dim[i] == 1)
    return dims


def _valid_squeeze_args(a, dims):
    return utils.is_of_tensor(a) and utils.is_dims_arg(dims)


def unsqueeze(a, dims):
    assert _valid_squeeze_args(a, dims)
    out = funcs.Unsqueeze.apply(a, dims)
    return out


def reshape(a, dim):
    assert _valid_reshape_args(a, dim)
    out = funcs.Reshape.apply(a, dim)
    return out


def _valid_reshape_args(a, dim):
    return utils.is_of_tensor(a) and utils.is_dims_arg(dim)


def clone(a):
    assert utils.is_of_tensor(a)
    out = funcs.Clone.apply(a)
    return out
