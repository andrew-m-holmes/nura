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


class _Embedding(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, w: Tensor, padid: Optional[int]):
        context.save(x, w)
        context["padid"] = padid
        return w.data[x.data]

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x, w = context.tensors()[0]
        padid = context["padid"]
        mask = np.zeros_like(w.data)
        mask[x.data] = 1
        if padid is not None:
            mask[padid] = 0
        return mask * grad.data
