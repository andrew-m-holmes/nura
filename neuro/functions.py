import numpy as np
from .tensors import Tensor
from .autograd.function import Context, Function
from neuro.types import dim
from typing import Any


class Add(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save(a, b)
        arr = a.data + b.data
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        return grad.data.copy(), grad.data.copy()

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        arr = agrad.data + bgrad.data
        return arr


class Sub(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save(a, b)
        arr = a.data - b.data
        return arr

    @staticmethod
    def backward(context: Any, grad: Tensor):
        return grad.data.copy(), np.negative(grad.data)

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        arr = agrad.data + np.negative(bgrad.data)
        return arr


class Mul(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save(a, b)
        arr = a.data * b.data
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        arr0 = b.data * grad.data
        arr1 = a.data * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr = agrad.data * b.data + bgrad.data * a.data
        return arr


class Div(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save(a, b)
        arr = a.data / b.data
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        arr0 = grad.data / b.data
        arr1 = np.negative(a.data) / b.data**2.0 * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr0 = agrad.data / b.data
        arr1 = a.data * (np.negative(bgrad.data) / b.data**2)
        arr = arr0 + arr1
        return arr


class Dot(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save(a, b)
        arr = np.dot(a.data, b.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        if a.ndim == 1 and b.ndim > 1:
            arr0 = np.dot(b.data, grad.data)
            arr1 = np.outer(a.data, grad.data)
        elif b.ndim == 1 and a.ndim > 1:
            arr0 = np.outer(grad.data, b.data)
            arr1 = np.dot(a.data.T, grad.data)
        else:
            arr0 = np.dot(grad.data, b.data.T)
            arr1 = np.dot(a.data.T, grad.data)
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr0 = np.dot(agrad.data, b.data)
        arr1 = np.dot(a.data, bgrad.data)
        arr = arr0 + arr1
        return arr


class Matmul(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save(a, b)
        arr = np.matmul(a.data, b.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        arr0 = np.matmul(grad.data, np.swapaxes(b.data, -2, -1))
        arr1 = np.matmul(np.swapaxes(a.data, -2, -1), grad.data)
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr0 = np.matmul(agrad.data, b.data)
        arr1 = np.matmul(a.data, bgrad.data)
        arr = arr0 + arr1
        return arr


class Pow(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        arr = np.power(a.data, b.data)
        context.save(a, b)
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        arr = context["arr"]
        arr0 = b.data * np.power(a.data, b.data - 1.0) * grad.data
        arr1 = arr * np.log(a.data) * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr = context["arr"]
        arr0 = b.data * np.power(a.data, b.data - 1.0) * agrad.data
        arr1 = np.log(a.data) * arr * bgrad.data
        return arr0 + arr1


class Exp(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        arr = np.exp(a.data)
        context.save(a)
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        return arr * grad.data

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        arr = context["arr"]
        return arr * agrad.data


class Log(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        arr = np.log(a.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = 1.0 / a.data * grad.data
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        a = context.tensors()[0]
        arr = 1.0 / a.data * agrad.data
        return arr


class Sin(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        arr = np.sin(a.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data * np.cos(a.data)
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        a = context.tensors()[0]
        arr = np.cos(a.data) * agrad.data
        return arr


class Cos(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        arr = np.cos(a.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data * np.negative(np.sin(a.data))
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        a = context.tensors()[0]
        arr = agrad.data * np.negative(np.sin(a.data))
        return arr


class Sum(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dim, keepdims: bool):
        context.save(a)
        context["dim"] = dim
        context["keepdims"] = keepdims
        arr = np.sum(a.data, dim, keepdims=keepdims)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context["dim"]
        keepdims = context["keepdims"]
        graddata = grad.data
        if not keepdims:
            graddata = np.expand_dims(graddata, axis=dim)
        arr = np.ascontiguousarray(np.broadcast_to(graddata, a.dim))
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = np.sum(agrad.data, axis=dim, keepdims=keepdims)
        return arr


class Max(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: int, keepdims: bool):
        context.save(a)
        context["dim"] = dim
        context["keepdims"] = keepdims
        arr = np.max(a.data, dim, keepdims=True)
        context["arr"] = arr
        if not keepdims:
            arr = np.squeeze(arr)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = context["arr"]
        graddata = grad.data
        mask = a.data == arr
        if not keepdims:
            graddata = np.expand_dims(graddata, axis=dim)
        arr = np.ascontiguousarray(np.broadcast_to(graddata, a.dim))
        return mask * arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        a = context.tensors()[0]
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = context["arr"]
        mask = a.data == arr
        graddata = np.where(mask, agrad.data, -np.inf)
        return np.max(graddata, axis=dim, keepdims=keepdims)


class Min(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: int, keepdims: bool):
        context.save(a)
        context["dim"] = dim
        context["keepdims"] = keepdims
        arr = np.min(a.data, dim, keepdims=True)
        context["arr"] = arr
        if not keepdims:
            arr = np.squeeze(arr)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = context["arr"]
        graddata = grad.data
        mask = a.data == arr
        if not keepdims:
            graddata = np.expand_dims(graddata, axis=dim)
        arr = np.ascontiguousarray(np.broadcast_to(graddata, a.dim))
        return mask * arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        a = context.tensors()[0]
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = context["arr"]
        mask = a.data == arr
        graddata = np.where(mask, agrad.data, np.inf)
        return np.min(graddata, axis=dim, keepdims=keepdims)


class Squeeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dim):
        context.save(a)
        context["dim"] = dim
        arr = a.data.squeeze(axis=dim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim = context["dim"]
        arr = np.expand_dims(grad.data, axis=dim)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        dim = context["dim"]
        arr = grad.data.squeeze(axis=dim)
        return arr


class Unsqueeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: Any):
        context.save(a)
        context["dim"] = dim
        arr = np.expand_dims(a.data, axis=dim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim = context["dim"]
        arr = grad.data.squeeze(axis=dim)
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        dim = context["dim"]
        arr = np.expand_dims(agrad.data, axis=dim)
        return arr


class View(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim):
        context.save(a)
        context["dim"] = dim
        arr = a.data.reshape(dim, order="C")
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data.reshape(a.dim, order="C")
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        dim = context["dim"]
        arr = agrad.data.reshape(dim, order="C")
        return arr


class Reshape(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: Any):
        context.save(a)
        context["dim"] = dim
        arr = a.data.reshape(dim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data.reshape(a.dim)
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        dim = context["dim"]
        arr = agrad.data.reshape(dim)
        return arr


class Transpose(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim0: int, dim1: int):
        arr = a.data.swapaxes(dim0, dim1)
        context.save(a)
        context["dim0"] = dim0
        context["dim1"] = dim1
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim0 = context["dim0"]
        dim1 = context["dim1"]
        arr = grad.data.swapaxes(dim0, dim1)
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        dim0 = context["dim0"]
        dim1 = context["dim1"]
        arr = agrad.data.swapaxes(dim0, dim1)
        return arr


class Permute(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim):
        context.save(a)
        context["dim"] = dim
        arr = a.data.transpose(dim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim = np.argsort(context["dim"])
        arr = grad.data.transpose(dim)
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        dim = context["dim"]
        arr = agrad.data.transpose(dim)
        return arr


class Abs(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        return np.absolute(a.data)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        mask = np.where(a.data < 0, -1, 1)
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        return abs(agrad.data)


class Clone(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        arr = a.data.copy()
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        return grad.data.copy()

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        arr = agrad.data.copy()
        return arr


class Slice(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, slc: slice):
        context.save(a)
        context["slc"] = slc
        context["dim"] = a.dim
        arr = a.data[slc]
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        slc = context["slc"]
        mask = np.zeros_like(a.data)
        mask[slc] = grad.data
        arr = mask
        return arr

    @staticmethod
    def tangent(context: Context, agrad: Tensor):
        slc = context["slc"]
        arr = agrad.data[slc]
        return arr
