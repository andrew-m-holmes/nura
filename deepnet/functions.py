import numpy as np
from .tensors import Tensor
from .autograd.function import Context, Function
from deepnet.types import _dim
from typing import Any


class Add(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        arr = a.data + b.data
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        return grad.data, grad.data

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor, bgrad: Tensor):
        arr = agrad.data + bgrad.data
        return arr


class Sub(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        arr = a.data - b.data
        return arr

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return grad.data, np.negative(grad.data)

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor, bgrad: Tensor):
        arr = agrad.data + np.negative(bgrad.data)
        return arr


class Mul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        arr = a.data * b.data
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        arr0 = b.data * grad.data
        arr1 = a.data * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor, bgrad: Tensor):
        a, b = ctx.tensors()
        arr = agrad.data * b.data + bgrad.data * a.data
        return arr


class Div(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        arr = a.data / b.data
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        arr0 = grad.data / b.data
        arr1 = np.negative(a.data) / b.data**2.0 * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor, bgrad: Tensor):
        a, b = ctx.tensors()
        arr0 = agrad.data / b.data
        arr1 = a.data * (np.negative(bgrad.data) / b.data**2)
        arr = arr0 + arr1
        return arr


class Dot(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        arr = np.dot(a.data, b.data)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        arr0 = np.dot(grad.data, b.data)
        arr1 = np.dot(a.data, grad.data)
        return arr0, arr1

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor, bgrad: Tensor):
        a, b = ctx.tensors()
        arr0 = np.dot(agrad.data, b.data)
        arr1 = np.dot(a.data, bgrad.data)
        arr = arr0 + arr1
        return arr


class Matmul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        arr = a.data @ b.data
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        arr0 = grad.data @ np.swapaxes(b.data, -2, -1)
        arr1 = np.swapaxes(a.data, -2, -1) @ grad.data
        return arr0, arr1

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor, bgrad: Tensor):
        a, b = ctx.tensors()
        arr0 = agrad.data @ b.data
        arr1 = a.data @ bgrad.data
        arr = arr0 + arr1
        return arr


class Pow(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        arr = np.power(a.data, b.data)
        ctx.save(a, b)
        ctx["arr"] = arr
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        arr = ctx["arr"]
        arr0 = b.data * np.power(a.data, b.data - 1.0) * grad.data
        arr1 = arr * np.log(a.data) * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor, bgrad: Tensor):
        a, b = ctx.tensors()
        arr = ctx["arr"]
        arr0 = b.data * np.power(a.data, b.data - 1.0) * agrad.data
        arr1 = np.log(a.data) * arr * bgrad.data
        return arr0 + arr1


class Exp(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        arr = np.exp(a.data)
        ctx.save(a)
        ctx["arr"] = arr
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        arr = ctx["arr"]
        return arr * grad.data

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        arr = ctx["arr"]
        return arr * agrad.data


class Log(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        arr = np.log(a.data)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        arr = 1.0 / a.data * grad.data
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        a = ctx.tensors()[0]
        arr = 1.0 / a.data * agrad.data
        return arr


class Sin(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        arr = np.sin(a.data)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        arr = grad.data * np.cos(a.data)
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        a = ctx.tensors()[0]
        arr = np.cos(a.data) * agrad.data
        return arr


class Cos(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        arr = np.cos(a.data)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        arr = grad.data * np.negative(np.sin(a.data))
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        a = ctx.tensors()[0]
        arr = agrad.data * np.negative(np.sin(a.data))
        return arr


class Sum(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: _dim, keepdims: bool):
        ctx.save(a)
        ctx["dim"] = dim
        ctx["keepdims"] = keepdims
        arr = np.sum(a.data, dim, keepdims=keepdims)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        dim = ctx["dim"]
        keepdims = ctx["keepdims"]
        graddata = grad.data
        if not keepdims:
            graddata = np.expand_dims(graddata, axis=dim)
        arr = np.ascontiguousarray(np.broadcast_to(graddata, a.dim))
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        dim = ctx["dim"]
        keepdims = ctx["keepdims"]
        arr = np.sum(agrad.data, axis=dim, keepdims=keepdims)
        return arr


class Squeeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: _dim):
        ctx.save(a)
        ctx["dim"] = dim
        arr = a.data.squeeze(axis=dim)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = ctx["dim"]
        arr = np.expand_dims(grad.data, axis=dim)
        return arr

    @staticmethod
    def tangent(ctx: Context, grad: Tensor):
        dim = ctx["dim"]
        arr = grad.data.squeeze(axis=dim)
        return arr


class Unsqueeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx["dim"] = dim
        arr = np.expand_dims(a.data, axis=dim)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = ctx["dim"]
        arr = grad.data.squeeze(axis=dim)
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        dim = ctx["dim"]
        arr = np.expand_dims(agrad.data, axis=dim)
        return arr


class View(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx["dim"] = dim
        arr = a.data.reshape(dim, order="C")
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        arr = grad.data.reshape(a.dim, order="C")
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        dim = ctx["dim"]
        arr = agrad.data.reshape(dim, order="C")
        return arr


class Reshape(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx["dim"] = dim
        arr = a.data.reshape(dim)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        arr = grad.data.reshape(a.dim)
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        dim = ctx["dim"]
        arr = agrad.data.reshape(dim)
        return arr


class Transpose(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim0: int, dim1: int):
        arr = a.data.swapaxes(dim0, dim1)
        ctx.save(a)
        ctx["dim0"] = dim0
        ctx["dim1"] = dim1
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim0 = ctx["dim0"]
        dim1 = ctx["dim1"]
        arr = grad.data.swapaxes(dim0, dim1)
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        dim0 = ctx["dim0"]
        dim1 = ctx["dim1"]
        arr = agrad.data.swapaxes(dim0, dim1)
        return arr


class Permute(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx["dim"] = dim
        arr = a.data.transpose(dim)
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = np.argsort(ctx["dim"])
        arr = grad.data.transpose(dim)
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        dim = ctx["dim"]
        arr = agrad.data.transpose(dim)
        return arr


class Abs(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        return np.absolute(a.data)

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        mask = np.where(a.data < 0, -1, 1)
        return mask * grad.data

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        return abs(agrad.data)


class Clone(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        arr = a.data.copy()
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        return grad.data

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        arr = agrad.data.copy()
        return arr


class Slice(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, _slice: slice):
        ctx.save(a)
        ctx["slice"] = _slice
        ctx["dim"] = a.dim
        arr = a.data[_slice]
        return arr

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()[0]
        _slice = ctx["slice"]
        mask = np.zeros_like(a.data)
        mask[_slice] = grad.data
        arr = mask
        return arr

    @staticmethod
    def tangent(ctx: Context, agrad: Tensor):
        _slice = ctx["slice"]
        arr = agrad.data[_slice]
        return arr
