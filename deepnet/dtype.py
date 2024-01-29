import numpy as np
import deepnet

_py_int = int
_py_float = float
_py_bool = bool


class dtype:

    _can_diff = None
    _wrapping = None

    @classmethod
    def can_diff(cls):
        return cls._can_diff

    @classmethod
    def numpy(cls, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data, cls._wrapping)
        if np.dtype(data.dtype) is not np.dtype(cls._wrapping):
            data = data.astype(cls._wrapping)
        return data

    @classmethod
    def name(cls):
        return cls.__name__


class byte(dtype):

    _can_diff = False
    _wrapping = np.uint8


class char(dtype):

    _can_diff = False
    _wrapping = np.int8


class short(dtype):

    _can_diff = False
    _wrapping = np.int16


class int(dtype):

    _can_diff = False
    _wrapping = np.int32


class long(dtype):

    _can_diff = False
    _wrapping = np.int64


class half(dtype):

    _can_diff = True
    _wrapping = np.float16


class float(dtype):

    _can_diff = True
    _wrapping = np.float32


class double(dtype):

    _can_diff = True
    _wrapping = np.float64


class bool(dtype):

    _can_diff = False
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


def to(obj, dtype):
    deepnet.istensor(obj)
    data = dtype.numpy(obj.data)
    return deepnet.tensor(data, obj.diff, dtype)

def typename(obj):
    assert deepnet.istensor(obj)
    obj_dtype = obj.dtype
    return obj_dtype.name().capitalize() + str(obj.__class__.__name__)


def get_dtype(data) -> dtype:
    if isinstance(data, np.ndarray):
        return _dtype_map[data.dtype]
    if isinstance(data, list):
        return get_dtype(np.array(data))
    dtype = type(data)
    assert dtype in _dtype_map
    return _dtype_map[dtype]


