import numpy as np
import deepnet
from deepnet.dtype import dtype
from deepnet import Tensor


def zeros(dim, use_grad=False, dtype=None):
    dim = preprocess_dim(dim)
    zero_arr = np.zeros(dim)
    return deepnet.tensor(zero_arr, use_grad, dtype)


def zeros_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return deepnet.tensor(zero_arr, use_grad, dtype)


def ones(dim, use_grad=False, dtype=None):
    dim = preprocess_dim(dim)
    ones_arr = np.ones(dim)
    return deepnet.tensor(ones_arr, use_grad, dtype)


def ones_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data)
    return deepnet.tensor(ones_arr, use_grad, dtype)


def randn(dim=None, use_grad=False, dtype=None):
    dim = preprocess_dim(dim)
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr, use_grad, dtype)


def rand(dim=None, use_grad=False, dtype=None):
    dim = preprocess_dim(dim)
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr, use_grad, dtype)


def randint(low, high, dim, dtype=None):
    dim = preprocess_dim(dim)
    randint_arr = np.random.randint(low, high, dim)
    return deepnet.tensor(randint_arr, dtype)


def identity(n, use_grad=False, dtype=None):
    data = np.identity(n)
    return deepnet.tensor(data, use_grad, dtype)


def full(dim, num, use_grad=False, dtype=None):
    dim = preprocess_dim(dim)
    data = np.full(dim, num)
    return deepnet.tensor(data, use_grad, dtype)


def preprocess_to_tensors(*items):
    assert all(is_tensor(item) or is_py_scalar(item)
               for item in items)
    processed_items = tuple(
        deepnet.tensor(item) if not isinstance(
            item, Tensor) else item for item in items)
    return processed_items if len(
        processed_items) > 1 else processed_items[0]
    

def preprocess_dim(dim):
    if dim is None:
        return tuple()
    if is_py_scalar(dim):
        dim = (dim,)
    return dim

def to_contiguous(tensor):
    if is_contiguous(tensor):
        return tensor
    contiguous_tensor = tensor.clone()
    contiguous_tensor.data = np.ascontiguousarray(tensor.data)
    return contiguous_tensor

def is_all_tensor(*items):
    return all(is_tensor(item) for item in items)


def is_tensor(item):
    return isinstance(item, Tensor)


def is_contiguous(tensor):
    return tensor.data.flags["C_CONTIGUOUS"]


def is_numpy(item):
    numpy_types = [
        np.ndarray, np.uint8, np.int8, np.int16, np.int32, np.int64,
        np.float16, np.float32, np.float64, np.bool_]
    return type(item) in numpy_types


def is_all_py_scalars(*items):
    return all(is_py_scalar(item) for item in items)


def is_py_scalar(item):
    py_scalar_types = [float, int]
    return type(item) in py_scalar_types


def is_py_bool(item):
    return isinstance(item, bool)


def is_py_list(item):
    return isinstance(item, list)


def is_dims_arg(arg):
    if arg is None or isinstance(arg, int):
        return True
    return all(is_py_scalar(val) for val in arg)


def is_scalar_tensor(item):
    if is_tensor(item):
        return item.dim() == 0
    return False


def is_dtype(item):
    return issubclass(item, dtype)
