import numpy as np
import deepnet
from typing import Tuple

class TensorBase:

    def __init__(self, data, dtype) -> None:
        self.data = data
        self.dtype = dtype

    def dim(self) -> Tuple[int, ...]:
        return self.data.shape

    def ndim(self) -> int:
        return self.data.ndim

    def nelem(self) -> int:
        return self.data.size

    def item(self):
        assert self.nelem() == 1
        return self.data.item()

    def to(self, dtype):
        return _to(self, dtype)

    def byte(self):
        return self.to(deepnet.byte)

    def char(self):
        return self.to(deepnet.char)

    def short(self):
        return self.to(deepnet.short)

    def int(self):
        return self.to(deepnet.int)

    def long(self):
        return self.to(deepnet.long)

    def half(self):
        return self.to(deepnet.half)

    def float(self):
        return self.to(deepnet.float)

    def double(self):
        return self.to(deepnet.double)

    def bool(self):
        return self.to(deepnet.bool)

    def clone(self):
        return deepnet.clone(self)

    def contiguous(self):
        return deepnet.to_contiguous(self)

    def sum(self, dims=None, keepdims=False):
        return deepnet.sum(self, dims, keepdims)

    def squeeze(self, dims=None):
        return deepnet.squeeze(self, dims)

    def unsqueeze(self, dims):
        return deepnet.unsqueeze(self, dims)

    def view(self, dim):
        return deepnet.view(self, dim)

    def reshape(self, dim):
        return deepnet.reshape(self, dim)

    def transpose(self, dim_0=-2, dim_1=-1):
        return deepnet.transpose(self, dim_0, dim_1)

    def permute(self, dims=None):
        return deepnet.permute(self, dims=dims)

    def withattr(self, **attrs):
        return _withattrs(self, **attrs)
    
    def __add__(self, other):
        return deepnet.add(self, other)

    def __radd__(self, other):
        return deepnet.add(self, other)

    def __sub__(self, other):
        return deepnet.sub(self, other)
    
    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
        return deepnet.mul(self, -1.0)

    def __abs__(self):
        raise NotImplementedError

    def __rsub__(self, other):
        return deepnet.sub(other, self)

    def __mul__(self, other):
        return deepnet.mul(self, other)

    def __rmul__(self, other):
        return deepnet.mul(self, other)

    def __truediv__(self, other):
        return deepnet.div(self, other)

    def __rtruediv__(self, other):
        return deepnet.div(other, self)

    def __matmul__(self, other):
        return deepnet.matmul(self, other)

    def __rmatmul__(self, other):
        return deepnet.matmul(other, self)

    def __pow__(self, other):
        return deepnet.pow(self, other)

    def __rpow__(self, other):
        return deepnet.pow(other, self)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, _slice):
        return deepnet.slice(self, _slice)

    def __setitem__(self, _slice, item):
        self.data[_slice] = item.data if deepnet.is_tensor(item) else item

    def __repr__(self) -> str:
        base = repr(self.data)
        rep = base.replace("array", "tensor").replace(",", "").replace(")", "")
        if " dtype" in rep:
            start = rep.index(" dtype")
            rep = rep[:start]
        return rep + ")"

class Tensor(TensorBase):

    def __init__(self, data, diff, dtype) -> None:
        super().__init__(data, dtype)
        self.diff = diff
        self.grad = None
        self.backfn = None
        self.leaf = True

    def backward(self, grad=None):
        raise NotImplementedError

    def dual(self, tan=None):
        return dual(self, tan)

class DualTensor(TensorBase):
    
    def __init__(self, data, tan, dtype) -> None:
        super().__init__(data, dtype)
        self.tan = tan

    def unpack(self):
        return tensor(self.data, False, self.dtype), self.tan

    def __repr__(self) -> str:
        return super().__repr__().replace("tensor", "  dual")

def tensor(data, diff=False, dtype=None):
    if dtype is None:
        dtype = deepnet.get_dtype(data)
    diff = dtype.can_diff and diff
    data = dtype.numpy(data)
    return Tensor(data, diff, dtype)

# def _handle_tensor(data, diff, dtype):
#     if (not deepnet.is_py_scalar(data) or not deepnet.is_py_bool(data) 
#         or not deepnet.is_py_list(data) or not isinstance(data, np.ndarray))

def dual(primal, tan=None):
    # _handle_dual(primal, tan)
    if tan is None:
        tan = deepnet.zeros_like(primal)
    return DualTensor(primal.data, tan, primal.dtype)

# def _handle_dual(primal, tan):
#     if not isinstance(primal, Tensor):
#         raise ValueError(f"Invalid argument for primal: {primal}")
#     if not primal.dtype.can_diff():
#         raise TypeError("dtype of primal is not differentiable valid dtypes=[deepnet.half, deepnet.float, deepnet.double]")
#     if tan is not None:
#         if not isinstance(tan, Tensor):
#             raise ValueError(f"Invalid argument for tan: {tan}")
#         if primal.dtype != tan.dtype:
#             raise TypeError(f"dtype mismatch between primal ({primal.dtype.name()}) and tan ({tan.dtype.name()})")
#         if primal.dim() != tan.dim():
#             raise ValueError(f"dim mismatch between primal ({primal.dim()}) and tan ({tan.dim()})")
#         if tan.diff:
#             raise AttributeError("tan cannot be a differentiable Tensor")

def _to(obj, dtype):
    if isinstance(obj, Tensor):
        data = dtype.numpy(obj.data)
        return Tensor(data, obj.diff, dtype)
    elif isinstance(obj, DualTensor):
        data = dtype.numpy(obj.data)
        tan = tan.to(dtype)
        return DualTensor(data, tan, dtype)

def _withattrs(obj, **attrs):
    if isinstance(obj, Tensor):
        obj = tensor(obj.data, obj.diff, obj.dtype)
    elif isinstance(obj, DualTensor):
        obj = dual(obj, obj.tan)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj