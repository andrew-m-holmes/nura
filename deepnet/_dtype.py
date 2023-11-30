import numpy as np
import deepnet


class _dtype:

    _differentiable = None
    _wrapping = None

    @classmethod
    def differentiable(cls):
        return cls._differentiable

    @classmethod
    def cast(cls, tensor):
        data = tensor.data.astype(cls._wrapping)
        casted = deepnet.tensor(data, use_grad=tensor.use_grad, dtype=cls)
        return casted

    @classmethod
    def name(cls):
        return cls.__name__


class byte(_dtype):

    _differentiable = False
    _wrapping = np.uint8


class char(_dtype):

    _differentiable = False
    _wrapping = np.int8


class short(_dtype):

    _differentiable = False
    _wrapping = np.int16


class int(_dtype):

    _differentiable = False
    _wrapping = np.int32


class long(_dtype):

    _differentiable = False
    _wrapping = np.int64


class half(_dtype):

    _differentiable = True
    _wrapping = np.float16


class float(_dtype):

    _differentiable = True
    _wrapping = np.float32


class double(_dtype):

    _differentiable = True
    _wrapping = np.float64


class bool(_dtype):

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
