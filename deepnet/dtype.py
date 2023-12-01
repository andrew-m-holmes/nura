import numpy as np
import deepnet


class dtype:

    _differentiable = None
    _wrapping = None

    @classmethod
    def differentiable(cls):
        return cls._differentiable

    @classmethod
    def cast(cls, tensor):
        data = tensor.data.astype(cls._wrapping)
        new_tensor = deepnet.tensor(
            data, tensor.use_grad, tensor.dtype)
        return new_tensor

    @classmethod
    def name(cls):
        return cls.__name__


class byte(dtype):

    _differentiable = False
    _wrapping = np.uint8


class char(dtype):

    _differentiable = False
    _wrapping = np.int8


class short(dtype):

    _differentiable = False
    _wrapping = np.int16


class int(dtype):

    _differentiable = False
    _wrapping = np.int32


class long(dtype):

    _differentiable = False
    _wrapping = np.int64


class half(dtype):

    _differentiable = True
    _wrapping = np.float16


class float(dtype):

    _differentiable = True
    _wrapping = np.float32


class double(dtype):

    _differentiable = True
    _wrapping = np.float64


class bool(dtype):

    _differentiable = False
    _wrapping = np.bool_


dtype_map = {
    np.uint8: byte,
    np.int8: char,
    np.int16: short,
    np.int32: int,
    np.int64: long,
    np.float16: half,
    np.float32: float,
    np.float64: double,
    np.bool_: bool
}


def infer_dtype(data):
    if isinstance(data, np.ndarray):
        return dtype_map.get(data.dtype)
    elif isinstance(data, list):
        return _infer_dtype_from_list(data)
    return dtype_map.get(type(data))


def _infer_dtype_from_list(data):
    for item in data:
        if isinstance(item, list):
            nested_type = _infer_dtype_from_list(item)
            if nested_type is not None:
                return nested_type
        return infer_dtype(item)
    return None
