import numpy as np
import importlib
import deepnet
import deepnet.utils as utils
from typing import Tuple

_nn_func = "deepnet.nn.functional"


class Tensor:

    def __init__(self, data, use_grad, dtype) -> None:
        self.data = data
        self.grad = None
        self.grad_fn = None
        self.use_grad = use_grad
        self.is_leaf = True
        self.dtype = dtype

    def backward(self, grad=None):
        if grad is None:
            grad = utils.ones_like(self, use_grad=False)
        self.grad_fn.apply(grad)

    def dim(self) -> Tuple[int, ...]:
        return self.data.shape

    def ndim(self) -> int:
        return self.data.ndim

    def detach(self) -> "Tensor":
        return tensor(self, False, self.dtype)

    def clone(self) -> "Tensor":
        f = _import_module(_nn_func)
        return f.clone(self)

    def zero(self):
        self.grad = utils.zeros_like(self, use_grad=False)

    def squeeze(self, dim=None) -> "Tensor":
        f = _import_module(_nn_func)
        return f.squeeze(self, dim)

    def tranpose(self, dim_0, dim_1) -> "Tensor":
        f = _import_module(_nn_func)
        return f.tranpose(self, dim_0, dim_1)

    def _set_grad_state(self, use_grad, grad_fn, is_leaf):
        self.use_grad = use_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf

    def __repr__(self) -> str:
        rep = f"tensor({self.data}, "
        rep += f"use_grad={self.use_grad}"
        if self.use_grad:
            rep += f", grad_fn={self.grad_fn}"
        rep += f", dtype={self.dtype.name()}"
        rep += ")"
        return rep

    def __add__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.mul(self, other)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.div(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.matmul(self, other)

    def __pow__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.pow(self, other)


class DualTensor:

    def __init__(self, primal, tangent=None) -> None:
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
        f = _import_module(_nn_func)
        primal = f.clone(self.primal)
        tangent = f.clone(self.tangent)
        return DualTensor(primal, tangent)

    def _set_grad_state(self, use_grad, grad_fn, is_leaf):
        self.use_grad = use_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf

    def __repr__(self) -> str:
        rep = f"dual_tensor(primal: {self.primal}, tangent: {self.tangent})"
        return rep

    def __add__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.mul(self, other)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.div(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.matmul(self, other)

    def __pow__(self, other: "Tensor") -> "Tensor":
        f = _import_module(_nn_func)
        return f.pow(self, other)


def tensor(data, use_grad=False, dtype=None):
    # TODO implement preprocess
    # data, dtype = _preprocess_tensor_init(data, use_grad, dtype)
    return Tensor(data, use_grad, dtype)


def dual_tensor(primal, tangent=None):
    # TODO implement preprocess
    # tangent = _preprocess_dual_tensor_init(primal, tangent)
    return DualTensor(primal, tangent)


def _import_module(name):
    try:
        module = importlib.import_module(name)
        return module
    except:
        raise ValueError(f"Unknown module name: {name}")
