import numpy as np
import deepnet
from .tensors import Tensor
from .autograd.function import Context, Function
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
        grad_b = deepnet.tensor(-1. * a.data / b.data **
                                2. * grad.data)
        return grad_a, grad_b


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


class Sum(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims, keepdims):
        context.save_tensors(a)
        context.a_dim = a.dim()
        out = deepnet.tensor(np.sum(a.data, dims, keepdims=keepdims))
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a_dim = context.a_dim
        grad_out = deepnet.tensor(np.ascontiguousarray(
            np.broadcast_to(
                grad.data, a_dim)))
        return grad_out


class Squeeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims: int):
        context.save_tensors(a)
        context.dims = dims
        out = deepnet.tensor(a.data.squeeze(axis=dims))
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dims = context.dims
        grad_out = deepnet.tensor(
            np.expand_dims(grad.data, axis=dims))
        return grad_out


class Unsqueeze(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims: int):
        context.save_tensors(a)
        out = deepnet.tensor(
            np.expand_dims(a.data, axis=dims),
            dtype=a.dtype)
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        return deepnet.tensor(grad.data.squeeze())


class Reshape(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim: int):
        context.save_tensors(a)
        context.a_dim = a.dim()
        out = deepnet.tensor(a.data.reshape(dim))
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a_dim = context.a_dim
        return deepnet.tensor(grad.data.reshape(a_dim))


class Tranpose(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dim_0: int, dim_1: int):
        out = deepnet.tensor(a.data.swapaxes(dim_0, dim_1))
        context.save_tensors(a)
        context.dim_0 = dim_0
        context.dim_1 = dim_1
        return out

    @staticmethod
    def backward(context: Context, grad: Tensor):
        dim_0 = context.dim_0
        dim_1 = context.dim_1
        grad_out = deepnet.tensor(grad.data.swapaxes(dim_0, dim_1))
        return grad_out


class Permute(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, dims):
        context.save_tensors(a)
        context.dims = dims
        out = deepnet.tensor(a.data.transpose(dims))
        return out

    def backward(context: Context, grad: Tensor):
        dims = np.argsort(context.dims)
        grad_out = deepnet.tensor(grad.data.transpose(dims))
        return grad_out


class Clone(Function):

    @staticmethod
    def forward(context: Context, a: Tensor):
        context.save_tensors(a)
        out = deepnet.tensor(
            a.data.copy(), use_grad=a.use_grad)
        return out

    @staticmethod
    def backward(context: Any, grad: Tensor):
        return grad
