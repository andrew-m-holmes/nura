import numpy as np
from .tensors import Tensor
from .autograd.function import Context, Function
from typing import Any


class Add(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data + b.data
        return out

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        del ctx
        return grad.data, grad.data

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        tan = atan.data + btan.data
        return tan


class Sub(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data - b.data
        return out

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        del ctx
        return grad.data, np.negative(grad.data)

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        tan = atan.data + np.negative(btan.data)
        return tan


class Mul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data * b.data
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        agrad = b.data * grad.data
        bgrad = a.data * grad.data
        return agrad, bgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        a, b = ctx.tensors()
        tan = atan.data * b.data + btan.data * a.data
        return tan


class Div(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data / b.data
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        agrad = 1.0 / b.data * grad.data
        bgrad = np.negative(a.data) / b.data**2.0 * grad.data
        return agrad, bgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        a, b = ctx.tensors()
        atandata= atan.data / b.data
        btandata= a.data * (np.negative(btan.data) / b.data**2)
        tan = atandata + btandata
        return tan


# TODO needs tests
class Dot(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = np.dot(a.data, b.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        agrad = np.dot(grad.data, b.data)
        bgrad = np.dot(a.data, grad.data)
        return agrad, bgrad

    @staticmethod
    def tangent(ctx: Context):
        a, b = ctx.tensors()
        atan = np.dot(a.tangent, b.data)
        btan = np.dot(a.data, b.tangent)
        tan = atan + btan
        return tan


class Matmul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data @ b.data
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        agrad = grad.data @ np.swapaxes(b.data, -2, -1)
        bgrad = np.swapaxes(a.data, -2, -1) @ grad.data
        return agrad, bgrad

    @staticmethod
    def tangent(ctx: Context):
        a, b = ctx.tensors()
        atan = a.tangent.data @ b.data
        btan = a.data @ b.tangent.data
        tan = atan + btan
        return tan


class Pow(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        out = np.power(a.data, b.data)
        ctx.save(a, b)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        out = ctx.out
        agrad = b.data * np.power(a.data, b.data - 1.0) * grad.data
        bgrad = out * np.log(a.data) * grad.data
        return agrad, bgrad

    @staticmethod
    def tangent(ctx: Context):
        a, b = ctx.tensors()
        out = ctx.out
        atan = b.data * np.power(a.data, b.data - 1.0) * a.tangent.data
        btan = np.log(a.data) * out * b.tangent.data
        tan = atan + btan
        return tan


class Exp(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        out = np.exp(a.data)
        ctx.save(a)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        out = ctx.out
        outgrad = out * grad.data
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        out = ctx.out
        tan = out * a.tangent.data
        return tan


class Log(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = np.log(a.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        outgrad = 1.0 / a.data * grad.data
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        tan = 1.0 / a.data * a.tangent.data
        return tan


class Sine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = np.sin(a.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        agrad = grad.data * np.cos(a.data)
        return agrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        tan = np.cos(a.data) * a.tangent.data
        return tan


class Cosine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = np.cos(a.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        agrad = grad.data * np.negative(np.sin(a.data))
        return agrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        tan = a.tangent.data * np.negative(np.sin(a.data))
        return tan


class Sum(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim, keepdims):
        ctx.save(a)
        ctx.adim = a.dim()
        ctx.dim = dim
        ctx.keepdims = keepdims
        out = np.sum(a.data, dim, keepdims=keepdims)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        adim = ctx.adim
        dim = ctx.dim
        keepdims = ctx.keepdims
        graddata = grad.data
        if not keepdims:
            graddata = np.expand_dims(graddata, axis=dim)
        outgrad = np.ascontiguousarray(np.broadcast_to(graddata, adim))
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        dim = ctx.dim
        keepdims = ctx.keepdims
        tan = np.sum(a.tangent.data, axis=dim, keepdims=keepdims)
        return tan


class Squeeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: int):
        ctx.save(a)
        ctx.dim = dim
        out = a.data.squeeze(axis=dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = ctx.dim
        outgrad = np.expand_dims(grad.data, axis=dim)
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        dim = ctx.dim
        tan = a.tangent.data.squeeze(axis=dim)
        return tan


class Unsqueeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx.dim = dim
        out = np.expand_dims(a.data, axis=dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = ctx.dim
        outgrad = grad.data.squeeze(axis=dim)
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        dim = ctx.dim
        tan = np.expand_dims(a.tangent.data, axis=dim)
        return tan


class View(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx.adim = a.dim()
        ctx.dim = dim
        out = a.data.reshape(dim, order="C")
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        adim = ctx.adim
        outgrad = grad.data.reshape(adim, order="C")
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        dim = ctx.dim
        tan = a.tangent.data.reshape(dim, order="C")
        return tan


class Reshape(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx.adim = a.dim()
        ctx.dim = dim
        out = a.data.reshape(dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        adim = ctx.adim
        outgrad = grad.data.reshape(adim)
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        dim = ctx.dim
        tan = a.tangent.data.reshape(dim)
        return tan


class Tranpose(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim_0: int, dim_1: int):
        out = a.data.swapaxes(dim_0, dim_1)
        ctx.save(a)
        ctx.dim_0 = dim_0
        ctx.dim_1 = dim_1
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim_0 = ctx.dim_0
        dim_1 = ctx.dim_1
        outgrad = grad.data.swapaxes(dim_0, dim_1)
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        dim_0 = ctx.dim_0
        dim_1 = ctx.dim_1
        tan = a.tangent.data.swapaxes(dim_0, dim_1)
        return tan


class Permute(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx.dim = dim
        out = a.data.transpose(dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = np.argsort(ctx.dim)
        outgrad = grad.data.transpose(dim)
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors[0]
        dim = ctx.dim
        tan = a.tangent.data.transpose(dim)
        return tan


class Abs(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        return np.absolute(a.data)

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        mask = np.where(a.data < 0, -1, 1)
        return mask * grad.data

    @staticmethod
    def tangent(ctx: Context, tan: Tensor):
        del ctx
        return abs(tan.data)


class Clone(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = a.data.copy()
        return out

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return (grad,)

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors()
        tan = a.tangent.data.copy()
        return tan


class Slice(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, _slice: slice):
        ctx.save(a)
        ctx.slice = _slice
        ctx.dim = a.dim()
        out = a.data[_slice]
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        _slice = ctx.slice
        mask = np.zeros_like(a.data)
        mask[_slice] = grad.data
        outgrad = mask
        return outgrad

    @staticmethod
    def tangent(ctx: Context):
        a = ctx.tensors[0]
        _slice = ctx.slice
        tan = a.tangent.data[_slice]
        return tan
