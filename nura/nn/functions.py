import numpy as np
from nura.autograd.function import Function, Context
from nura.tensors import Tensor
from typing import Optional


class _ReLU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        return np.maximum(z.data, 0)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        mask = np.where(z.data > 0, 1, 0)
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        mask = np.where(z.data > 0, 1, 0)
        return mask * zgrad.data


class _ReLU6(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        return np.clip(z.data, 0, 6)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        mask = np.where((z.data > 0) & (z.data < 6), 1, 0)
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        mask = np.where((z.data > 0) & (z.data < 6), 1, 0)
        return mask * zgrad.data


class _LeakyReLU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, slope: float):
        context.save(z)
        context["slope"] = slope
        return np.maximum(z.data * slope, z.data)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        slope = context["slope"]
        mask = np.where(z.data >= 0, 1, slope)
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        slope = context["slope"]
        mask = np.where(z.data >= 0, 1, slope)
        return mask * zgrad.data


class _ELU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, alpha: float):
        context.save(z)
        context["alpha"] = alpha
        return np.where(z.data > 0, z.data, alpha * (np.exp(z.data) - 1))

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        alpha = context["alpha"]
        mask = np.where(z.data > 0, 1, alpha * np.exp(z.data))
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        alpha = context["alpha"]
        mask = np.where(z.data > 0, 1, alpha * np.exp(z.data))
        return mask * zgrad.data


class _CELU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, alpha: float):
        context.save(z)
        context["alpha"] = alpha
        arr0 = np.maximum(z.data, 0)
        arr1 = np.minimum(0, alpha * (np.exp(z.data / alpha) - 1))
        return arr0 + arr1

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        alpha = context["alpha"]
        mask = np.where(z.data >= 0, 1, np.exp(z.data / alpha))
        return mask * grad.data


class _Embedding(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, w: Tensor, padid: Optional[int]):
        context.save(w)
        context["xdata"] = x.data
        context["padid"] = padid
        return w.data[x.data]

    @staticmethod
    def backward(context: Context, grad: Tensor):
        w = context.tensors()[0]
        xdata = context["xdata"]
        padid = context["padid"]

        mask = xdata != padid
        indices = xdata[mask]
        grads = grad.data[mask]
        arr = np.zeros_like(w.data)
        np.add.at(arr, indices, grads)
        return arr
