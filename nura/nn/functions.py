import numpy as np
from nura.autograd.function import Function, Context
from nura.tensors import Tensor
from typing import Optional


class _Sigmoid(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        arr = 1 / (1 + np.exp(-z.data))
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        return arr * (1 - arr) * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        arr = context["arr"]
        return arr * (1 - arr) * zgrad.data


class _Tanh(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        arr = np.tanh(z.data)
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        return (1 - arr**2) * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        arr = context["arr"]
        return (1 - arr**2) * zgrad.data


class _Softmax(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, dim: int):
        context.save(z)
        e = np.exp(z.data)
        arr = e / e.sum(axis=dim, keepdims=True)
        context["arr"] = arr
        return arr

    # TODO FIX
    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        jacobian = np.diag(arr.flatten()) - np.outer(arr, arr)
        return np.reshape(jacobian, arr.shape + arr.shape[-1:])

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        arr = context["arr"]
        return (np.diagflat(arr) - np.outer(arr, arr)) * zgrad.data


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


class _GELU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        PICONST = 0.79788456
        CONST = 0.044715
        context["PICONST"] = PICONST
        context["CONST"] = CONST

        tanh = np.tanh(PICONST * (z.data + CONST * np.power(z.data, 3)))
        inner = tanh + 1.0
        context["tanh"] = tanh
        context["inner"] = inner
        return 0.5 * z.data * inner

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        PICONST = context["PICONST"]
        CONST = context["CONST"]
        tanh = context["tanh"]
        inner = context["inner"]
        dtanh = 1 - tanh**2
        dgelu = 0.5 * (
            inner + z.data * PICONST * dtanh * (1 + 3 * CONST * np.power(z.data, 2))
        )
        return dgelu * grad.data


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
