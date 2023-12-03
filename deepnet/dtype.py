import numpy as np
import deepnet

_py_int = int
_py_float = float
_py_bool = bool


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
    def numpy(cls, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return data.astype(cls._wrapping)

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


_dtype_map = {
    np.uint8: byte,
    np.int8: char,
    np.int16: short,
    np.int32: int,
    _py_int: int,
    np.int64: long,
    np.float16: half,
    np.float32: float,
    _py_float: float,
    np.float64: double,
    np.bool_: bool,
    _py_bool: bool,
    np.dtype(np.uint8): byte,
    np.dtype(np.int8): char,
    np.dtype(np.int16): short,
    np.dtype(np.int32): int,
    np.dtype(np.int64): long,
    np.dtype(np.float16): half,
    np.dtype(np.float32): float,
    np.dtype(np.float64): double,
    np.dtype(np.bool_): bool,
}


def _infer_dtype(data):
    if isinstance(data, np.ndarray):
        return _dtype_map.get(data.dtype)
    if isinstance(data, list):
        return _dtype_map.get(_infer_dtype_from_list(data))
    return _dtype_map.get(type(data))


def _infer_dtype_from_list(data):
    if isinstance(data, list):
        return _infer_dtype_from_list(data[0])
    return type(data)
