import numpy as np
import deepnet
from deepnet import Tensor
from deepnet.autograd.function import Function, Context
from typing import Any


class Add(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save_tensors(a, b)
        out = deepnet.tensor(a.data + b.data)
        return out

    @staticmethod
    def backward(context: Any, grad: Tensor):
        grad_a = deepnet.tensor(1. * grad.data)
        grad_b = deepnet.tensor(1. * grad.data)
        return grad_a, grad_b

    @staticmethod
    def jvp(context: Any, tangent_a: Tensor, tangent_b: Tensor):
        tangent_out = deepnet.tensor(tangent_a.data + tangent_b.data)
        return tangent_out


def add(a, b):
    out = Add.apply(a, b)
    return out


class Sub(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save_tensors(a, b)
        out = deepnet.tensor(a.data - b.data)
        return out

    @staticmethod
    def backward(context: Any, grad: Tensor):
        grad_a = grad
        grad_b = deepnet.tensor(grad.data * -1.)
        return grad_a, grad_b

    @staticmethod
    def jvp(context: Any, tangent_a: Tensor, tangent_b: Tensor):
        tangent_out = deepnet.tensor(tangent_a.data - tangent_b.data)
        return tangent_out


def sub(a, b):
    out = Sub.apply(a, b)
    return out


class Mul(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save_tensors(a, b)
        out = deepnet.tensor(a.data * b.data)
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.saved_tensors()
        grad_a = deepnet.tensor(b.data * grad.data)
        grad_b = deepnet.tensor(a.data * grad.data)
        return grad_a, grad_b

    @staticmethod
    def jvp(context: Context, tangent_a: Tensor, tangent_b: Tensor):
        a, b = context.saved_tensors()
        tangent_out = deepnet.tensor(
            tangent_a.data * b.data + tangent_b.data * a.data)
        return tangent_out


def mul(a, b):
    out = Mul.apply(a, b)
    return out


class Div(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save_tensors(a, b)
        out = deepnet.tensor(a.data / b.data)
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.saved_tensors()
        grad_a = deepnet.tensor(1. / b.data * grad.data)
        grad_b = deepnet.tensor(-1. * a.data / b.data ** 2. * grad.data)
        return grad_a, grad_b


def div(a, b):
    out = Div.apply(a, b)
    return out


class Matmul(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save_tensors(a, b)
        out = a.data @ b.data
        return deepnet.tensor(out)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.saved_tensors()
        dim_a, dim_b = np.arange(a.ndim()), np.arange(b.ndim())
        dim_a[-2], dim_a[-1] = dim_a[-1], dim_a[-2]
        dim_b[-2], dim_b[-1] = dim_b[-1], dim_b[-2]
        grad_a = deepnet.tensor(grad.data @ b.data.transpose(dim_b))
        grad_b = deepnet.tensor(a.data.transpose(dim_a) @ grad.data)
        return grad_a, grad_b


def matmul(a, b):
    out = Matmul.apply(a, b)
    return out


class Pow(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        out = deepnet.tensor(np.power(a.data, b.data))
        context.save_tensors(a, b, out)
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b, out = context.saved_tensors()
        grad_a = deepnet.tensor(
            b.data * np.power(a.data, b.data - 1.) * grad.data)
        grad_b = deepnet.tensor(out.data * np.log(a.data) * grad.data)
        return grad_a, grad_b


def power(a, b):
    out = Pow.apply(a, b)
    return out


class Sine(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save_tensors(a)
        out = deepnet.tensor(np.sin(a.data))
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.saved_tensors()[0]
        grad_a = deepnet.tensor(grad.data * np.cos(a.data))
        return grad_a


def sine(a):
    out = Sine.apply(a)
    return out


class Cosine(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save_tensors(a)
        out = deepnet.tensor(np.cos(a.data))
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context.saved_tensors()[0]
        grad_a = deepnet.tensor(grad.data * -1. * np.sin(a.data))
        return grad_a


def cosine(a):
    out = Cosine.apply(a)
    return out


class Squeeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: int):
        # TODO this needs to be fixed
        a_dim = a.dim()
        context.save_tensors(a)
        context.a_dim = a_dim
        out_dim = []
        for i, d in enumerate(a_dim):
            if d != 1 or i not in dim:
                out_dim.append(d)
        out_dim = tuple(out_dim)
        out = deepnet.tensor(a.data.reshape(out_dim))
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a_dim = context.a_dim
        grad_data = deepnet.tensor(grad.data.reshape(a_dim))
        return grad_data


def squeeze(a, dim):
    out = Squeeze.apply(a, dim=dim)
    return out


class Tranpose(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim_0: int, dim_1: int):
        size = np.arange(a.ndim())
        size[dim_0], size[dim_1] = size[dim_1], size[dim_0]
        out = deepnet.tensor(a.data.transpose(size))
        context.save_tensors(a)
        context.size = size
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        size = context.size
        grad_data = deepnet.tensor(grad.data.transpose(size))
        return grad_data


def tranpose(a, dim_0, dim_1):
    out = Tranpose.apply(a, dim_0=dim_0, dim_1=dim_1)
    return out


class Clone(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save_tensors(a)
        out = deepnet.tensor(
            a.data.copy(), use_grad=a.use_grad, dtype=a.dtype)
        return out

    @staticmethod
    def backward(context: Any, grad: Tensor):
        return grad


def clone(a):
    out = Clone.apply(a)
    return out
