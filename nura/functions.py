import numpy as np
from .tensors import Tensor
from .autograd.function import Context, Function
from nura.types import dim, dimlike
from typing import Any, Tuple, Union, Optional

np._set_promotion_state("weak")


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
        arr = a.data * (1 / b.data)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        arr0 = grad.data * (1 / b.data)
        arr1 = np.negative(a.data) * (1 / np.square(b.data)) * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr0 = agrad.data * (1 / b.data)
        arr1 = a.data * np.negative(1 / np.square(b.data)) * bgrad.data
        arr = arr0 + arr1
        return arr


class Floordiv(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        arr = a.data // b.data
        return arr


class Modulo(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        arr = a.data % b.data
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
        arr0 = b.data * grad.data
        arr1 = a.data * grad.data
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
        return np.matmul(a.data, b.data)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        if a.ndim == 1:
            axis = tuple(range(b.ndim - 2)) + (b.ndim - 1,)
            arr0 = (b.data * np.expand_dims(grad.data, -2)).sum(axis=axis)
            arr1 = np.einsum("...jl,k->...jkl", grad.data, a.data)
        elif b.ndim == 1:
            axis = tuple(range(a.ndim - 1))
            arr0 = np.einsum("...,l->...l", grad.data, b.data)
            arr1 = (a.data * np.expand_dims(grad.data, -1)).sum(axis=axis)
        else:
            arr1 = np.matmul(a.data.swapaxes(-2, -1), grad.data)
            arr0 = np.matmul(grad.data, b.data.swapaxes(-2, -1))
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
        context.arr = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        arr = context.arr
        arr0 = b.data * np.power(a.data, b.data - 1) * grad.data
        arr1 = arr * np.log(a.data) * grad.data
        return arr0, arr1

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        a, b = context.tensors()
        arr = context.arr
        arr0 = b.data * np.power(a.data, b.data - 1) * agrad.data
        arr1 = np.log(a.data) * arr * bgrad.data
        return arr0 + arr1


class Exp(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        arr = np.exp(a.data)
        context.save(a)
        context.arr = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context.arr
        return arr * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        arr = context.arr
        return arr * grad.data


class Log(Function):

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
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = np.cos(a.data) * grad.data
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
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data * np.negative(np.sin(a.data))
        return arr


class Sum(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike, keepdims: bool):
        context.save(a)
        context.dim = dim
        context.keepdims = keepdims
        arr = np.sum(a.data, dim, keepdims=keepdims)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context.dim
        keepdims = context.keepdims
        graddata = grad.data
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return graddata + np.zeros_like(a.data)

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        dim = context.dim
        keepdims = context.keepdims
        arr = np.sum(grad.data, axis=dim, keepdims=keepdims)
        return arr


class Max(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike, keepdims: bool):
        context.save(a)
        arr = np.max(a.data, dim, keepdims=True)
        context.dim = dim
        context.keepdims = keepdims
        context.arr = arr
        if not keepdims:
            arr = np.squeeze(arr)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context.dim
        keepdims = context.keepdims
        arr = context.arr
        graddata = grad.data
        mask = a.data == arr
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return mask * (graddata + np.zeros_like(a.data))

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context.dim
        keepdims = context.keepdims
        arr = context.arr
        mask = a.data == arr
        graddata = np.where(mask, grad.data, -np.inf)
        return np.max(graddata, axis=dim, keepdims=keepdims)


class Min(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike, keepdims: bool):
        context.save(a)
        arr = np.min(a.data, dim, keepdims=True)
        context.dim = dim
        context.keepdims = keepdims
        context.arr = arr
        if not keepdims:
            arr = np.squeeze(arr)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context.dim
        keepdims = context.keepdims
        arr = context.arr
        graddata = grad.data
        mask = a.data == arr
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return mask * (graddata + np.zeros_like(a.data))

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context.dim
        keepdims = context.keepdims
        arr = context.arr
        mask = a.data == arr
        graddata = np.where(mask, grad.data, np.inf)
        return np.min(graddata, axis=dim, keepdims=keepdims)


class Mean(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike, keepdims: bool):
        context.save(a)
        context.dim = dim
        context.keepdims = keepdims
        return a.data.mean(axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        dim = context.dim
        keepdims = context.keepdims
        graddata = grad.data
        n = np.prod(a.dim) if dim is None else np.prod(a.dim[dim])
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return graddata * (1 / n)


class Var(Function):

    @staticmethod
    def forward(
        context: Context, a: Tensor, correction: int, dim: dimlike, keepdims: bool
    ):
        context.save(a)
        context.correction = correction
        context.dim = dim
        context.keepdims = keepdims
        mean = a.data.mean(axis=dim, keepdims=keepdims)
        context.mean = mean
        return a.data.var(ddof=correction, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        correction = context.correction
        dim = context.dim
        keepdims = context.keepdims
        mean = context.mean
        graddata = grad.data

        n = (
            np.prod(a.dim)
            if dim is None
            else np.prod(tuple(a.dim[i] for i in range(a.ndim))) - correction
        )
        if not keepdims and a.data.shape != graddata.shape:
            graddata = np.expand_dims(graddata, axis=dim)
        return graddata * (2 / n) * -mean


class Squeeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: Optional[dimlike]):
        context.save(a)
        context.dim = dim
        arr = a.data.squeeze(axis=dim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim = context.dim
        arr = np.expand_dims(grad.data, axis=dim)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        dim = context.dim
        arr = grad.data.squeeze(axis=dim)
        return arr


class Unsqueeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: dimlike):
        context.save(a)
        context.dim = dim
        arr = np.expand_dims(a.data, axis=dim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim = context.dim
        arr = grad.data.squeeze(axis=dim)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        dim = context.dim
        arr = np.expand_dims(grad.data, axis=dim)
        return arr


class Reshape(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, newdim: dim):
        context.save(a)
        context.newdim = newdim
        arr = a.data.reshape(newdim)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        arr = grad.data.reshape(a.dim)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        newdim = context.newdim
        arr = grad.data.reshape(newdim)
        return arr


class Transpose(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim0: int, dim1: int):
        arr = a.data.swapaxes(dim0, dim1)
        context.save(a)
        context.dim0 = dim0
        context.dim1 = dim1
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim0 = context.dim0
        dim1 = context.dim1
        arr = grad.data.swapaxes(dim0, dim1)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        dim0 = context.dim0
        dim1 = context.dim1
        arr = grad.data.swapaxes(dim0, dim1)
        return arr


class Permute(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims: dim):
        context.save(a)
        context.dims = dims
        context.ndim = a.ndim
        arr = a.data.transpose(dims)
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        normed = tuple(i + context.ndim if i < 0 else i for i in (context.dims))
        dims = np.argsort(normed)
        arr = grad.data.transpose(dims)
        return arr

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        normed = tuple(i + context.ndim if i < 0 else i for i in (context.dims))
        dims = np.argsort(normed)
        arr = grad.data.transpose(dims)
        return arr


class Abs(Function):

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


class Pos(Function):

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


class Neg(Function):

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
    def tangent(context: Context, grad: Tensor):
        return grad.data.copy()


class Slice(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, slice_: Union[Tuple[slice, ...], slice]):
        context.save(a)
        context.slice_ = slice_
        arr = a.data[slice_]
        return arr.copy()

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.tensors()[0]
        slice_ = context.slice_
        mask = np.zeros_like(a.data)
        mask[slice_] = grad.data
        return mask

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        slice_ = context.slice_
        arr = grad.data[slice_]
        return arr.copy()


class Flatten(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, start: int, end: int):
        context.save(a)
        dim = a.dim
        if end < 0:
            end += a.ndim
        newdim = dim[:start] + (np.prod(dim[start : end + 1]),)
        if end < a.ndim - 1:
            newdim += dim[end + 1 :]
        context.dim = dim
        context.newdim = newdim
        return a.data.reshape(newdim)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim = context.dim
        return grad.data.reshape(dim)

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        newdim = context.newdim
        return grad.data.reshape(newdim)


class Concat(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor, dim: int):
        context.save(a, b)
        context.dim = dim
        return np.concatenate((a.data, b.data), axis=dim)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.tensors()
        dim = context.dim
        index = a.data.shape[dim]
        output = tuple(np.split(grad.data, (index, index), axis=dim))
        return (output[0].copy(), output[-1].copy())

    @staticmethod
    def tangent(context: Context, agrad: Tensor, bgrad: Tensor):
        dim = context.dim
        return np.concatenate((agrad.data, bgrad.data), axis=dim)
