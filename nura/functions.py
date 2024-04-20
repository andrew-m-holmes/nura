import numpy as np
from .tensors import Tensor
from .autograd.function import Context, Function
from nura.types import dim, dimlike
from typing import Any

np._set_promotion_state("weak")


class _Add(Function):

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


class _Sub(Function):

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


class _Mul(Function):

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


class _Div(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save(a, b)
        arr = a.data * (1 / b.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        arr0 = grad.data * (1 / b.data)
        arr1 = a.data * np.negative(1 / np.square(b.data)) * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr0 = agrad.data * (1 / b.data)
        arr1 = a.data * np.negative(1 / np.square(b.data)) * bgrad.data
        arr = arr0 + arr1
        return arr


class _Floordiv(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        arr = a.data // b.data
        return arr


class _Modulo(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        arr = a.data % b.data
        return arr


class _Dot(Function):

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


class _Matmul(Function):

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


class _Pow(Function):

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
        arr0 = b.data * np.power(a.data, b.data - 1) * grad.data
        arr1 = arr * np.log(a.data) * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr = context["arr"]
        arr0 = b.data * np.power(a.data, b.data - 1) * agrad.data
        arr1 = np.log(a.data) * arr * bgrad.data
        return arr0 + arr1


class _Exp(Function):

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
    def tangent(context: Context, grad: Tensor):
        arr = context["arr"]
        return arr * grad.data


class _Log(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        arr = np.log(a.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = (1 / a.data) * grad.data
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = (1 / a.data) * grad.data
        return arr


class _Sin(Function):

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
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = np.cos(a.data) * grad.data
        return arr


class _Cos(Function):

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
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data * np.negative(np.sin(a.data))
        return arr


class _Sum(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike, keepdims: bool):
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
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return graddata + np.zeros_like(a.data)

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = np.sum(grad.data, axis=dim, keepdims=keepdims)
        return arr


class _Max(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike, keepdims: bool):
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
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return mask * (graddata + np.zeros_like(a.data))

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = context["arr"]
        mask = a.data == arr
        graddata = np.where(mask, grad.data, -np.inf)
        return np.max(graddata, axis=dim, keepdims=keepdims)


class _Min(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike, keepdims: bool):
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
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return mask * (graddata + np.zeros_like(a.data))

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context["dim"]
        keepdims = context["keepdims"]
        arr = context["arr"]
        mask = a.data == arr
        graddata = np.where(mask, grad.data, np.inf)
        return np.min(graddata, axis=dim, keepdims=keepdims)


class _Squeeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike):
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


class _Unsqueeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike):
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
    def tangent(context: Context, grad: Tensor):
        dim = context["dim"]
        arr = np.expand_dims(grad.data, axis=dim)
        return arr


class _View(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, newdim: dim):
        context.save(a)
        context["newdim"] = newdim
        arr = a.data.reshape(newdim, order="C")
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data.reshape(a.dim, order="C")
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        newdim = context["newdim"]
        arr = grad.data.reshape(newdim, order="C")
        return arr


class _Reshape(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, newdim: dim):
        context.save(a)
        context["newdim"] = newdim
        arr = a.data.reshape(newdim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data.reshape(a.dim)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        newdim = context["newdim"]
        arr = grad.data.reshape(newdim)
        return arr


class _Transpose(Function):

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
    def tangent(context: Context, grad: Tensor):
        dim0 = context["dim0"]
        dim1 = context["dim1"]
        arr = grad.data.swapaxes(dim0, dim1)
        return arr


class _Permute(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims: dim):
        context.save(a)
        context["dims"] = dims
        arr = a.data.transpose(dims)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dims = np.argsort(context["dims"])
        arr = grad.data.reshape(a.data.shape).transpose(dims)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dims = context["dims"]
        arr = grad.data.reshape(a.data.shape).transpose(dims)
        return arr


class _Abs(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        return np.absolute(a.data)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        mask = np.sign(a.data)
        return grad.data * mask

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        mask = np.sign(a.data)
        return grad.data * mask


class _Pos(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        return a.data.copy()

    @staticmethod
    def backward(context: Context, grad: Tensor):
        return grad.data.copy()

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        return grad.data.copy()


class _Neg(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        return np.negative(a.data)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        return np.negative(grad.data)

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        return np.negative(grad.data)


class _Clone(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save(a)
        arr = a.data.copy()
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        return grad.data.copy()

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        return grad.data.copy()


class _Slice(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, slc: slice):
        context.save(a)
        context["slc"] = slc
        arr = a.data[slc]
        return arr.copy()

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        slc = context["slc"]
        mask = np.zeros_like(a.data)
        mask[slc] = grad.data
        return mask

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        slc = context["slc"]
        arr = grad.data[slc]
        return arr.copy()
