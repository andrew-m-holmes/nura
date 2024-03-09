import numpy as np
from neuro.autograd.function import Function, Context
from neuro.tensors import Tensor


class ReLU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        return np.where(z.data > 0, z.data, 0.0)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        mask = np.where(z.data > 0, 1.0, 0.0)
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        mask = np.where(z.data > 0, 1.0, 0.0)
        return mask * zgrad.data


class LeakyReLU(Function):
    pass


class GeLU(Function):
    pass
