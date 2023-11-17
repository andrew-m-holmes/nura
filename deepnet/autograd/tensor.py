import numpy as np


class Tensor:

    def __init__(self, data, use_grad=False, dtype=None) -> None:
        self.data = np.array(data, dtype=dtype)
        self.grad = None
        self.grad_fn = None
        self.use_grad = use_grad
        self.is_leaf = True

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        self.grad_fn.apply(grad)

    def size(self):
        return self.data.shape

    def ndim(self):
        return self.data.ndim

    def dtype(self):
        return self.data.dtype

    def detach(self):
        return Tensor(self.data, use_grad=False)

    def tranpose(self, dim_0, dim_1):
        f = _functional_module()
        return f.tranpose(self, dim_0, dim_1)

    def __repr__(self) -> str:
        rep = f"({self.data}, "
        rep += f"grad_fn={self.grad_fn})" if self.use_grad \
            else f"use_grad={self.use_grad})"
        return rep

    def __add__(self, other: "Tensor") -> "Tensor":
        f = _functional_module()
        return f.add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        f = _functional_module()
        return f.sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        f = _functional_module()
        return f.mul(self, other)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        f = _functional_module()
        return f.div(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        f = _functional_module()
        return f.matmul(self, other)

    def __pow__(self, other: "Tensor") -> "Tensor":
        f = _functional_module()
        return f.pow(self, other)


def tensor(data, use_grad=False, dtype=None):
    dtypes = [
        int, float, list, np.ndarray, np.int8, np.int16, np.int32,
        np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64, np.float128,
    ]
    assert use_grad in [True, False], \
        f"use_grad only accepts bools not: {use_grad}"
    assert type(data) in dtypes, \
        f"Invalid object passed to tensor(): {type(data)}"
    if dtype is not None:
        assert dtype in dtypes, \
            f"Invalid dtype passed to tensor(): {dtype}"
    if isinstance(data, list):
        assert _list_checker(data, dtypes), \
            "Invalid data found in list, tensor() can only handle numeric dtypes"
    if isinstance(data, np.ndarray):
        assert data.dtype in dtypes, \
            "Invalid data found in numpy.ndarray, tensor() can only handle numeric dtypes"
    return Tensor(data, bool(use_grad), dtype)


def _list_checker(data, dtypes):
    if isinstance(data, list):
        return all(_list_checker(d, dtypes) for d in data)
    return type(data) in dtypes


def _functional_module():
    import deepnet.nn.functional as f
    return f