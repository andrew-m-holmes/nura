import numpy as np
import importlib
from typing import Tuple


class Tensor:

    def __init__(self, data, use_grad=False, dtype=None) -> None:
        self.data = _preprocess_tensor_init(data, use_grad, dtype)
        self.grad = None
        self.grad_fn = None
        self.use_grad = use_grad
        self.is_leaf = True

    def backward(self, grad=None):
        assert self.grad_fn is not None, \
            "Cannot differentiate a non-differentiable Tensor"
        if grad is None:
            utils = _import_module("deepnet.utils")
            grad = utils.ones_like(self, use_grad=False)
        self.grad_fn.apply(grad)

    @property
    def dtype(self):
        return self.data.dtype

    def dim(self) -> Tuple[int, ...]:
        return self.data.shape

    def ndim(self) -> int:
        return self.data.ndim

    def detach(self) -> "Tensor":
        return Tensor(self.data, use_grad=False)

    def clone(self) -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.clone(self)

    def zero(self):
        assert self.grad is not None, \
            "cannot zero a grad that does not exist"
        utils = _import_module("deepnet.utils")
        self.grad = utils.zeros_like(self, use_grad=False)

    def squeeze(self, dim=None) -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.squeeze(self, dim)

    def tranpose(self, dim_0, dim_1) -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.tranpose(self, dim_0, dim_1)

    def __repr__(self) -> str:
        rep = f"tensor({self.data}, "
        rep += f"use_grad={self.use_grad}"
        if self.use_grad:
            rep += f", grad_fn={self.grad_fn}"
        rep += ")"
        return rep

    def __add__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.mul(self, other)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.div(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.matmul(self, other)

    def __pow__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.pow(self, other)

    def _set_grad_state(self, use_grad, grad_fn, is_leaf):
        self.use_grad = use_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf


def tensor(data, use_grad=False, dtype=None):
    return Tensor(data, use_grad, dtype)


def _preprocess_tensor_init(data, use_grad, dtype=None):
    return np.array(data)


def _list_checker(data, dtypes):
    if isinstance(data, list):
        return all(_list_checker(d, dtypes) for d in data)
    return type(data) in dtypes


class DualTensor:

    def __init__(self, primal, tangent=None) -> None:
        tangent = _preprocess_dual_tensor_init(primal, tangent)
        self.primal = primal
        self.tangent = tangent

    @property
    def data(self):
        return self.primal.data

    @property
    def dtype(self):
        return self.primal.dtype

    def unpack(self) -> Tuple[Tensor, Tensor]:
        return self.primal, self.tangent

    def clone(self) -> "DualTensor":
        f = _import_module("deepnet.nn.functional")
        primal = f.clone(self.primal)
        tangent = f.clone(self.tangent)
        return DualTensor(primal, tangent)

    def __repr__(self) -> str:
        rep = f"dual_tensor(primal: {self.primal}, tangent: {self.tangent})"
        return rep

    def __add__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.mul(self, other)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.div(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.matmul(self, other)

    def __pow__(self, other: "Tensor") -> "Tensor":
        f = _import_module("deepnet.nn.functional")
        return f.pow(self, other)

    def _set_grad_state(self, use_grad, grad_fn, is_leaf):
        self.use_grad = use_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf


def dual_tensor(primal, tangent=None):
    return DualTensor(primal, tangent)


def _preprocess_dual_tensor_init(primal, tangent):
    assert isinstance(primal, Tensor), \
        f"Invalid argument passed to primal for DualTensor.__init__(): {primal}"
    if tangent is None:
        utils = _import_module("utils")
        tangent = utils.ones_like(primal, use_grad=False)
    assert isinstance(tangent, Tensor), \
        f"Invalid argument passed to tangent for DualTensor.__init__(): {tangent}"
    assert primal.dim() == tangent.dim(), \
        f"Dimension mismatch between primal and tangent: {primal.dim()} != {tangent.dim()}"
    return tangent


def _import_module(name):
    try:
        module = importlib.import_module(name)
        return module
    except:
        raise ValueError(f"Unknown module name: {name}")


def _infer_dtype(data):
    if isinstance(data, list):
        return _infer_dtype(data[0])
    if isinstance(data, np.ndarray):
        _dtype = _import_module("deepnet._dtype")
        dtype = data.dtype
        return _dtype.dtype_map[dtype]
    deepnet = _import_module("deepnet")
    dtype = type(data)
    if dtype is int:
        return deepnet.int
    elif dtype is float:
        return deepnet.float
    elif dtype is bool:
        return deepnet.bool
