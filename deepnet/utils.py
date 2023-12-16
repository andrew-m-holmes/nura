import numpy as np
import deepnet
from deepnet.dtype import dtype
from deepnet import Tensor, DualTensor


def zeros(dim, use_grad=False, dtype=None):
    zero_arr = np.zeros(dim)
    return deepnet.tensor(zero_arr, use_grad, dtype)


def zeros_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    zero_arr = np.zeros_like(data)
    return deepnet.tensor(zero_arr, use_grad, dtype)


def ones(dim, use_grad=False, dtype=None):
    ones_arr = np.ones(dim)
    return deepnet.tensor(ones_arr, use_grad, dtype)


def ones_like(tensor, use_grad=False, dtype=None):
    data = tensor.data
    ones_arr = np.ones_like(data)
    return deepnet.tensor(ones_arr, use_grad, dtype)


def randn(dim=None, use_grad=False, dtype=None):
    if dim is None:
        dim = ()
    randn_arr = np.random.randn(*dim)
    return deepnet.tensor(randn_arr, use_grad, dtype)


def rand(dim=None, use_grad=False, dtype=None):
    if dim is None:
        dim = ()
    rand_arr = np.random.rand(*dim)
    return deepnet.tensor(rand_arr, use_grad, dtype)


def randint(low, high, dim, dtype=None):
    randint_arr = np.random.randint(low, high, dim)
    return deepnet.tensor(randint_arr, dtype)


def identity(n, use_grad=False, dtype=None):
    data = np.identity(n)
    return deepnet.tensor(data, use_grad, dtype)


def full(dim, num, use_grad=False, dtype=None):
    data = np.full(dim, num)
    return deepnet.tensor(data, use_grad, dtype)


def is_of_tensor(*items):
    return all(is_tensor(item) or is_dual_tensor(item)
               for item in items)


def is_tensor(item):
    return isinstance(item, Tensor)


def is_dual_tensor(item):
    return isinstance(item, DualTensor)


def preprocess_to_tensors(*items):
    assert all(is_of_tensor(item) or is_py_scalar(item)
               for item in items)
    tensor_cls = [Tensor, DualTensor]
    processed_items = tuple(deepnet.tensor(item)
                            if type(item) not in tensor_cls else item
                            for item in items)
    return processed_items if len(
        processed_items) > 1 else processed_items[0]


def is_all_py_scalars(*items):
    return all(is_py_scalar(item) for item in items)


def is_py_scalar(item):
    py_scalar_types = [float, int]
    return type(item) in py_scalar_types


def is_numpy(item):
    numpy_types = [
        np.ndarray, np.uint8, np.int8, np.int16, np.int32, np.int64,
        np.float16, np.float32, np.float64, np.bool_]
    return type(item) in numpy_types


def is_py_bool(item):
    return isinstance(item, bool)


def is_py_list(item):
    return isinstance(item, list)


def is_dims_arg(arg):
    if arg is None or isinstance(arg, int):
        return True
    return all(is_py_scalar(val) for val in arg)


def is_scalar_tensor(item):
    assert is_of_tensor(item)
    return item.dim() == 0


def is_dtype(item):
    return issubclass(item, dtype)
