import numpy as np
from .tensors import Tensor
from .autograd.function import Context, Function
from typing import Any


class Add(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        rawout = a.data + b.data
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        del ctx
        return grad.data, grad.data

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        rawtan = atan.data + btan.data
        return rawtan


class Sub(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        rawout = a.data - b.data
        return rawout

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        del ctx
        return grad.data, np.negative(grad.data)

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        rawtan = atan.data + np.negative(btan.data)
        return rawtan


class Mul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        rawout = a.data * b.data
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        rawagrad = b.data * grad.data
        rawbgrad = a.data * grad.data
        return rawagrad, rawbgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        a, b = ctx.tensors()
        rawtan = atan.data * b.data + btan.data * a.data
        return rawtan


class Div(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        rawout = a.data / b.data
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        rawagrad = 1.0 / b.data * grad.data
        rawbgrad = np.negative(a.data) / b.data**2.0 * grad.data
        return rawagrad, rawbgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        a, b = ctx.tensors()
        rawatan = atan.data / b.data
        rawbtan = a.data * (np.negative(btan.data) / b.data**2)
        rawtan = rawatan + rawbtan
        return rawtan


class Dot(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        rawout = np.dot(a.data, b.data)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        rawagrad = np.dot(grad.data, b.data)
        rawbgrad = np.dot(a.data, grad.data)
        return rawagrad, rawbgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        a, b = ctx.tensors()
        rawatan = np.dot(atan.data, b.data)
        rawbtan = np.dot(a.data, btan.data)
        rawtan = rawatan + rawbtan
        return rawtan


class Matmul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        rawout = a.data @ b.data
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        rawagrad = grad.data @ np.swapaxes(b.data, -2, -1)
        rawbgrad = np.swapaxes(a.data, -2, -1) @ grad.data
        return rawagrad, rawbgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        a, b = ctx.tensors()
        rawatan = atan.data @ b.data
        rawbtan = a.data @ btan.data
        rawtan = rawatan + rawbtan
        return rawtan


class Pow(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        rawout = np.power(a.data, b.data)
        ctx.save(a, b)
        ctx["rawout"] = rawout
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        rawout = ctx["rawout"]
        rawagrad = b.data * np.power(a.data, b.data - 1.0) * grad.data
        rawbgrad = rawout * np.log(a.data) * grad.data
        return rawagrad, rawbgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor, btan: Tensor):
        a, b = ctx.tensors()
        rawout = ctx["rawout"]
        rawatan = b.data * np.power(a.data, b.data - 1.0) * atan.data
        rawbtan = np.log(a.data) * rawout * btan.data
        rawtan = rawatan + rawbtan
        return rawtan


class Exp(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        rawout = np.exp(a.data)
        ctx.save(a)
        ctx["rawout"] = rawout
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        rawout = ctx["rawout"]
        rawoutgrad = rawout * grad.data
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        rawout = ctx["rawout"]
        rawtan = rawout * atan.data
        return rawtan


class Log(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        rawout = np.log(a.data)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        rawoutgrad = 1.0 / a.data * grad.data
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        a = ctx.tensors()
        rawtan = 1.0 / a.data * atan.data
        return rawtan


class Sine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        rawout = np.sin(a.data)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        rawagrad = grad.data * np.cos(a.data)
        return rawagrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        a = ctx.tensors()
        rawtan = np.cos(a.data) * atan.data
        return rawtan


class Cosine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        rawout = np.cos(a.data)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        rawagrad = grad.data * np.negative(np.sin(a.data))
        return rawagrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        a = ctx.tensors()
        rawtan = atan.data * np.negative(np.sin(a.data))
        return rawtan


class Sum(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim, keepdims):
        ctx.save(a)
        ctx["dim"] = dim
        ctx["keepdims"] = keepdims
        rawout = np.sum(a.data, dim, keepdims=keepdims)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        dim = ctx["dim"]
        keepdims = ctx["keepdims"]
        graddata = grad.data
        if not keepdims:
            graddata = np.expand_dims(graddata, axis=dim)
        rawoutgrad = np.ascontiguousarray(np.broadcast_to(graddata, a.dim))
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        dim = ctx["dim"]
        keepdims = ctx["keepdims"]
        rawtan = np.sum(atan.data, axis=dim, keepdims=keepdims)
        return rawtan


class Squeeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: int):
        ctx.save(a)
        ctx["dim"] = dim
        rawout = a.data.squeeze(axis=dim)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = ctx["dim"]
        rawoutgrad = np.expand_dims(grad.data, axis=dim)
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        dim = ctx["dim"]
        rawtan = atan.data.squeeze(axis=dim)
        return rawtan


class Unsqueeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx["dim"] = dim
        rawout = np.expand_dims(a.data, axis=dim)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = ctx["dim"]
        rawoutgrad = grad.data.squeeze(axis=dim)
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        dim = ctx["dim"]
        rawtan = np.expand_dims(atan.data, axis=dim)
        return rawtan


class View(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx["dim"] = dim
        rawout = a.data.reshape(dim, order="C")
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        rawoutgrad = grad.data.reshape(a.dim, order="C")
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        dim = ctx["dim"]
        rawtan = atan.data.reshape(dim, order="C")
        return rawtan


class Reshape(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx["dim"] = dim
        rawout = a.data.reshape(dim)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        rawoutgrad = grad.data.reshape(a.dim)
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        dim = ctx["dim"]
        rawtan = atan.data.reshape(dim)
        return rawtan


class Tranpose(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim0: int, dim1: int):
        rawout = a.data.swapaxes(dim0, dim1)
        ctx.save(a)
        ctx["dim0"] = dim0
        ctx["dim1"] = dim1
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim0 = ctx["dim1"]
        dim1 = ctx["dim2"]
        rawoutgrad = grad.data.swapaxes(dim0, dim1)
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        dim0 = ctx["dim"]
        dim1 = ctx["dim"]
        rawtan = atan.data.swapaxes(dim0, dim1)
        return rawtan


class Permute(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx["dim"] = dim
        rawout = a.data.transpose(dim)
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim = np.argsort(ctx["dim"])
        rawoutgrad = grad.data.transpose(dim)
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        dim = ctx["dim"]
        rawtan = atan.data.transpose(dim)
        return rawtan


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
    def tangent(ctx: Context, atan: Tensor):
        del ctx
        return abs(atan.data)


class Clone(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        rawout = a.data.copy()
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        del ctx
        return grad.data

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        rawtan = atan.data.copy()
        return rawtan


class Slice(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, _slice: slice):
        ctx.save(a)
        ctx["slice"] = _slice
        ctx["dim"] = a.dim
        rawout = a.data[_slice]
        return rawout

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        _slice = ctx["slice"]
        mask = np.zeros_like(a.data)
        mask[_slice] = grad.data
        rawoutgrad = mask
        return rawoutgrad

    @staticmethod
    def tangent(ctx: Context, atan: Tensor):
        _slice = ctx["slice"]
        rawtan = atan.data[_slice]
        return rawtan
