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
        self.tangent = None
        self.in_dual = False

    def backward(self, grad=None):
        if grad is None:
            assert self.ndim() == 0
            grad = deepnet.ones_like(self, dtype=self.dtype)
        self.grad_fn.apply(grad)

    def dual(self, tangent=None, inplace=False):
        return make_dual(self, tangent, inplace)

    def undual(self):
        del self.tangent
        self._set_dual_state(None, False)

    def is_dual(self):
        return self.in_dual

    def dim(self) -> Tuple[int, ...]:
        return self.data.shape

    def ndim(self) -> int:
        return self.data.ndim

    def nelem(self):
        return self.data.size

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

    def detach(self):
        return tensor(self.data, False, dtype=self.dtype)

    def clone(self):
        return deepnet.clone(self)

    def zero(self):
        self.grad = deepnet.zeros_like(self, use_grad=False)

    def sum(self, dims, keepdims):
        return deepnet.sum(self, dims, keepdims)

    def squeeze(self, dims=None):
        return deepnet.squeeze(self, dims=dims)

    def unsqueeze(self, dims):
        return deepnet.unsqueeze(self, dims)

    def transpose(self, dim_0=-2, dim_1=-1):
        return deepnet.transpose(self, dim_0, dim_1)

    def contiguous(self):
        return deepnet.to_contiguous(self)

    def reshape(self, dim):
        return deepnet.reshape(self, dim)

    def _set_grad_state(self, use_grad, grad_fn, is_leaf):
        self.use_grad = use_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf

    def _set_dual_state(self, tangent, in_dual):
        self.tangent = tangent
        self.in_dual = in_dual

    def __repr__(self) -> str:
        rep = f"tensor({self.data}"
        if self.use_grad:
            rep += f", grad_fn={self.grad_fn}"
        else:
            rep += f", dtype={self.dtype.name()}"
        if self.in_dual:
            rep += f", in_dual=True"
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

    def __getitem__(self, indices):
        return tensor(self.data[indices],
                      self.use_grad, self.dtype)


def tensor(data, use_grad=False, dtype=None):
    _tensor_args_check(data, use_grad, dtype)
    data, dtype = _preprocess_tensor_args(data, use_grad, dtype)
    return Tensor(data, use_grad, dtype)


def _preprocess_tensor_args(data, use_grad, dtype):
    dtype = _infer_dtype(data) if dtype is None else dtype
    if use_grad:
        assert dtype.differentiable()
    data = dtype.numpy(data)
    return data, dtype


def _tensor_args_check(data, use_grad, dtype):
    assert _valid_tensor_data(data)
    assert deepnet.is_py_bool(use_grad)
    if dtype is not None:
        assert deepnet.is_dtype(dtype)

def _valid_tensor_data(data):
    return deepnet.is_numpy(data) or deepnet.is_py_list(
        data) or deepnet.is_py_scalar(data) or deepnet.is_py_bool(data)


def make_dual(tensor, tangent, inplace=False):
    _make_dual_args_check(tensor, tangent, inplace)
    tensor, tangent = _make_dual_helper(tensor, tangent, inplace)
    tensor._set_dual_state(tangent, True)
    return tensor


def _make_dual_helper(tensor, tangent, inplace):
    tangent = deepnet.zeros_like(
        tensor) if tangent is None else deepnet.preprocess_to_tensors(tangent)
    if inplace:
        return tensor, tangent
    if tensor.use_grad:
        return tensor.clone(), tangent
    return tensor.clone().detach(), tangent

def _make_dual_args_check(tensor, tangent, inplace):
    assert deepnet.is_tensor(tensor)
    if tangent is not None:
        assert deepnet.is_tensor(tangent)
        assert tensor.dim() == tangent.dim()
        assert tensor.dtype == tangent.dtype
    assert deepnet.is_py_bool(inplace)
