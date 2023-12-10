import numpy as np
import deepnet
from deepnet import Tensor
from deepnet.autograd.function import Function, Context
from typing import Any


class _Add(Function):

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


class _Sub(Function):

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


class _Mul(Function):

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


class _Div(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, b: Tensor):
        context.save_tensors(a, b)
        out = deepnet.tensor(a.data / b.data)
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, b = context.saved_tensors()
        grad_a = deepnet.tensor(1. / b.data * grad.data)
        grad_b = deepnet.tensor(-1. * a.data / b.data **
                                2. * grad.data)
        return grad_a, grad_b


class _Matmul(Function):

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


class _Pow(Function):

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


class _Sine(Function):

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


class _Cosine(Function):

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


class _Sum(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims=None,
                keepdims=False):
        context.save_tensors(a)
        out = deepnet.tensor(np.sum(a.data, dims, keepdims=keepdims))
        out_dim = out.dim()
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        # TODO figure out how to make the grad repeat to match the dims of input
        grad = deepnet.fill(grad.data)
        raise NotImplementedError


class _Squeeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims: int):
        context.save_tensors(a)
        context.dims = dims
        out = deepnet.tensor(a.data.squeeze(axis=dims))
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dims = context.dims
        grad_data = deepnet.tensor(
            np.expand_dims(grad.data, axis=dims))
        return grad_data


class _Unsqueeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims: int):
        pass

    @staticmethod
    def backward(context: Context, grad: Tensor):
        pass


class _Reshape(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims: int):
        pass

    @staticmethod
    def backward(context: Context, grad: Tensor):
        pass


def reshape(a, dims=None):
    pass


class _Tranpose(Function):

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


class _Clone(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save_tensors(a)
        out = deepnet.tensor(
            a.data.copy(), use_grad=a.use_grad, dtype=a.dtype)
        return out

    @staticmethod
    def backward(context: Any, grad: Tensor):
        return grad
