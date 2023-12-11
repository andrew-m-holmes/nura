import deepnet
from .dtype import _infer_dtype
from typing import Tuple


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
            assert self.nelem() == 1
            grad = deepnet.ones_like(self, dtype=self.dtype)
        self.grad_fn.apply(grad)

    def dim(self) -> Tuple[int, ...]:
        return self.data.shape

    def ndim(self) -> int:
        return self.data.ndim

    def nelem(self):
        return self.data.size

    def detach(self):
        return tensor(self.data, False, dtype=self.dtype)

    def byte(self):
        return tensor(self.data, False, dtype=deepnet.byte)

    def char(self):
        return tensor(self.data, False, dtype=deepnet.char)

    def short(self):
        return tensor(self.data, False, dtype=deepnet.short)

    def int(self):
        return tensor(self.data, False, dtype=deepnet.int)

    def long(self):
        return tensor(self.data, False, dtype=deepnet.long)

    def half(self):
        return tensor(self.data, self.use_grad, dtype=deepnet.half)

    def float(self):
        return tensor(self.data, self.use_grad, dtype=deepnet.float)

    def double(self):
        return tensor(self.data, self.use_grad, dtype=deepnet.double)

    def bool(self):
        return tensor(self.data, False, dtype=deepnet.bool)

    def clone(self):
        return deepnet.clone(self)

    def zero(self):
        self.grad = deepnet.zeros_like(self, use_grad=False)

    def sum(self, dims):
        raise NotImplementedError

    def squeeze(self, dims=None):
        return deepnet.squeeze(self, dims=dims)

    def transpose(self, dim_0=1, dim_1=0):
        return deepnet.transpose(self, dim_0, dim_1)

    def _set_grad_state(self, use_grad, grad_fn, is_leaf):
        self.use_grad = use_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf

    def __repr__(self) -> str:
        rep = f"tensor({self.data}"
        if self.use_grad:
            rep += f", grad_fn={self.grad_fn}"
        rep += f", dtype={self.dtype.name()}"
        rep += ")"
        return rep

    def __add__(self, other):
        return deepnet.add(self, other)

    def __sub__(self, other):
        return deepnet.sub(self, other)

    def __mul__(self, other):
        return deepnet.mul(self, other)

    def __truediv__(self, other):
        return deepnet.div(self, other)

    def __matmul__(self, other):
        return deepnet.matmul(self, other)

    def __pow__(self, other):
        return deepnet.pow(self, other)


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

    @property
    def use_grad(self):
        return self.primal.use_grad

    def unpack(self) -> Tuple[Tensor, Tensor]:
        return self.primal, self.tangent

    def clone(self):
        primal = deepnet.clone(self.primal)
        tangent = deepnet.clone(self.tangent)
        return dual_tensor(primal, tangent)

    def __repr__(self) -> str:
        rep = f"dual_tensor(primal: {self.primal}, tangent: {self.tangent})"
        return rep


def tensor(data, use_grad=False, dtype=None):
    data, dtype = _preprocess_tensor_args(data, use_grad, dtype)
    return Tensor(data, use_grad, dtype)


def _preprocess_tensor_args(data, use_grad, dtype):
    dtype = _infer_dtype(data) if dtype is None else dtype
    if use_grad:
        assert dtype.differentiable()
    data = dtype.numpy(data)
    return data, dtype


def dual_tensor(primal, tangent=None):
    tangent = _preprocess_dual_tensor_args(primal, tangent)
    return DualTensor(primal, tangent)


def _preprocess_dual_tensor_args(primal, tangent):
    assert deepnet.is_tensor(primal)
    if tangent is None:
        tangent = deepnet.ones_like(
            primal, use_grad=False, dtype=primal.dtype)
    assert deepnet.is_tensor(tangent)
    assert primal.dtype == tangent.dtype
    return tangent
