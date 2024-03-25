import numpy as np
from numpy import ndarray
from typing import Type, Any, Tuple, Union, Iterable


_py_int = int
_py_float = float
_py_bool = bool

inf = np.inf
dim = Tuple[int, ...]
dimlike = Union[Tuple[int, ...], int]
Scalar = Union[float, int]
Tensorlike = Union[Iterable[Any], Scalar, bool]


class dtype:

    _wrapping = None

    @classmethod
    def numpy(cls, data) -> ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data, cls._wrapping)
        if np.dtype(data.dtype) is not np.dtype(cls._wrapping):
            data = data.astype(cls._wrapping)
        return data

    @classmethod
    def name(cls) -> str:
        return cls.__name__


class byte(dtype):

    _wrapping = np.uint8


class char(dtype):

    _wrapping = np.int8


class short(dtype):

    _wrapping = np.int16


class int(dtype):

    _wrapping = np.int32


class long(dtype):

    _wrapping = np.int64


class half(dtype):

    _wrapping = np.float16


class float(dtype):

    _wrapping = np.float32


class double(dtype):

    _wrapping = np.float64


class bool(dtype):

    _wrapping = np.bool_


_dtypemap = {
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


def dtypeof(data: Any) -> Type[dtype]:
    if isinstance(data, np.ndarray):
        return _dtypemap[data.dtype]
    if isinstance(data, list) or isinstance(data, tuple):
        return dtypeof(data[0])
    dtype = type(data)
    if dtype not in _dtypemap:
        raise KeyError(f"Couldn't find {dtype} in dtype table")
    return _dtypemap[dtype]
