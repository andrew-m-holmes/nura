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
    def jvp(ctx: Context, tana, tanb):
        a, b = ctx.tensors()
        tan_out = a.tangent.data + b.tangent.data
        return tan_out


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
    def jvp(ctx: Context):
        a, b = ctx.tensors
        tan_b = np.negative(b.tangent.data)
        tan_out = a.tangent.data + tan_b
        return tan_out


class Mul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data * b.data
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        grad_a = b.data * grad.data
        grad_b = a.data * grad.data
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors()
        tan_a = b.data * a.tangent.data
        tan_b = a.data * b.tangent.data
        tan_out = tan_a + tan_b
        return tan_out


class Div(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data / b.data
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        grad_a = 1.0 / b.data * grad.data
        grad_b = np.negative(a.data) / b.data**2.0 * grad.data
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors()
        tan_a = a.tangent.data / b.data
        tan_b = a.data * (np.negative(b.tangent.data) / b.data**2)
        tan_out = tan_a + tan_b
        return tan_out


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
        grad_a = np.dot(grad.data, b.data)
        grad_b = np.dot(a.data, grad.data)
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors()
        tan_a = np.dot(a.tangent, b.data)
        tan_b = np.dot(a.data, b.tangent)
        tan_out = tan_a + tan_b
        return tan_out


class Matmul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = a.data @ b.data
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors()
        grad_a = grad.data @ np.swapaxes(b.data, -2, -1)
        grad_b = np.swapaxes(a.data, -2, -1) @ grad.data
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors()
        tan_a = a.tangent.data @ b.data
        tan_b = a.data @ b.tangent.data
        tan_out = tan_a + tan_b
        return tan_out


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
        grad_a = b.data * np.power(a.data, b.data - 1.0) * grad.data
        grad_b = out * np.log(a.data) * grad.data
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors()
        out = ctx.out
        tan_a = b.data * np.power(a.data, b.data - 1.0) * a.tangent.data
        tan_b = np.log(a.data) * out * b.tangent.data
        tan_out = tan_a + tan_b
        return tan_out


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
        grad_out = out * grad.data
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        out = ctx.out
        tan_out = out * a.tangent.data
        return tan_out


class Log(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = np.log(a.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        grad_out = 1.0 / a.data * grad.data
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        tan_out = 1.0 / a.data * a.tangent.data
        return tan_out


class Sine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = np.sin(a.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        grad_a = grad.data * np.cos(a.data)
        return grad_a

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        tan_out = np.cos(a.data) * a.tangent.data
        return tan_out


class Cosine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = np.cos(a.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors()
        grad_a = grad.data * np.negative(np.sin(a.data))
        return grad_a

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        tan_out = a.tangent.data * np.negative(np.sin(a.data))
        return tan_out


class Sum(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims, keepdims):
        ctx.save(a)
        ctx.a_dim = a.dim()
        ctx.dims = dims
        ctx.keepdims = keepdims
        out = np.sum(a.data, dims, keepdims=keepdims)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a_dim = ctx.a_dim
        dims = ctx.dims
        keepdims = ctx.keepdims
        grad_data = grad.data
        if not keepdims:
            grad_data = np.expand_dims(grad_data, axis=dims)
        grad_out = np.ascontiguousarray(np.broadcast_to(grad_data, a_dim))
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        dims = ctx.dims
        keepdims = ctx.keepdims
        tan_out = np.sum(a.tangent.data, axis=dims, keepdims=keepdims)
        return tan_out


class Squeeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims: int):
        ctx.save(a)
        ctx.dims = dims
        out = a.data.squeeze(axis=dims)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dims = ctx.dims
        grad_out = np.expand_dims(grad.data, axis=dims)
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        dims = ctx.dims
        tan_out = a.tangent.data.squeeze(axis=dims)
        return tan_out


class Unsqueeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims: Any):
        ctx.save(a)
        ctx.dims = dims
        out = np.expand_dims(a.data, axis=dims)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dims = ctx.dims
        grad_out = grad.data.squeeze(axis=dims)
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        dims = ctx.dims
        tan_out = np.expand_dims(a.tangent.data, axis=dims)
        return tan_out


class View(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx.a_dim = a.dim()
        ctx.dim = dim
        out = a.data.reshape(dim, order="C")
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a_dim = ctx.a_dim
        grad_out = grad.data.reshape(a_dim, order="C")
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        dim = ctx.dim
        tan_out = a.tangent.data.reshape(dim, order="C")
        return tan_out


class Reshape(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx.a_dim = a.dim()
        ctx.dim = dim
        out = a.data.reshape(dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a_dim = ctx.a_dim
        grad_out = grad.data.reshape(a_dim)
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        dim = ctx.dim
        tan_out = a.tangent.data.reshape(dim)
        return tan_out


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
        grad_out = grad.data.swapaxes(dim_0, dim_1)
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors()
        dim_0 = ctx.dim_0
        dim_1 = ctx.dim_1
        tan_out = a.tangent.data.swapaxes(dim_0, dim_1)
        return tan_out


class Permute(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims):
        ctx.save(a)
        ctx.dims = dims
        out = a.data.transpose(dims)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dims = np.argsort(ctx.dims)
        grad_out = grad.data.transpose(dims)
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dims = ctx.dims
        tan_out = a.tangent.data.transpose(dims)
        return tan_out


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
    def jvp(ctx: Context, tan: Tensor):
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
    def jvp(ctx: Context):
        a = ctx.tensors()
        tan_out = a.tangent.data.copy()
        return tan_out


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
        grad_out = mask
        return grad_out

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        _slice = ctx.slice
        tan_out = a.tangent.data[_slice]
        return tan_out
