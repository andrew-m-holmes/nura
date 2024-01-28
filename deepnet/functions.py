import numpy as np
import deepnet
from .tensors import Tensor
from .autograd.function import Context, Function
from typing import Any


class Add(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = deepnet.tensor(a.data + b.data)
        return out

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return grad, grad

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors
        tan_out = deepnet.tensor(a.tangent.data + b.tangent.data)
        return tan_out


class Sub(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = deepnet.tensor(a.data - b.data)
        return out

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        grad_a = grad
        grad_b = deepnet.tensor(np.negative(grad.data))
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors
        tan_b = np.negative(b.tangent.data)
        tan_out = deepnet.tensor(a.tangent.data + tan_b)
        return tan_out


class Mul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = deepnet.tensor(a.data * b.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors
        grad_a = deepnet.tensor(b.data * grad.data)
        grad_b = deepnet.tensor(a.data * grad.data)
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors
        tan_a = b.data * a.tangent.data
        tan_b = a.data * b.tangent.data
        tan_out = deepnet.tensor(tan_a + tan_b)
        return tan_out


class Div(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = deepnet.tensor(a.data / b.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors
        grad_a = deepnet.tensor(1. / b.data * grad.data)
        grad_b = deepnet.tensor(np.negative(a.data) / b.data ** 2. * grad.data)
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors
        tan_a = a.tangent.data / b.data
        tan_b = a.data * (np.negative(b.tangent.data) / b.data ** 2)
        tan_out = deepnet.tensor(tan_a + tan_b)
        return tan_out


# TODO needs tests
class Dot(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = deepnet.tensor(np.dot(a.data, b.data))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors
        grad_a = deepnet.tensor(np.dot(grad.data, b.data))
        grad_b = deepnet.tensor(np.dot(a.data, grad.data))
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors
        tan_a = np.dot(a.tangent, b.data)
        tan_b = np.dot(a.data, b.tangent)
        tan_out = deepnet.tensor(tan_a + tan_b)
        return tan_out


class Matmul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        ctx.save(a, b)
        out = deepnet.tensor(a.data @ b.data)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b = ctx.tensors
        grad_a = deepnet.tensor(grad.data @ np.swapaxes(b.data, -2, -1))
        grad_b = deepnet.tensor(np.swapaxes(a.data, -2, -1) @ grad.data)
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b = ctx.tensors
        tan_a = a.tangent.data @ b.data
        tan_b = a.data @ b.tangent.data
        tan_out = deepnet.tensor(tan_a + tan_b)
        return tan_out


class Pow(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor):
        out = deepnet.tensor(np.power(a.data, b.data))
        ctx.save(a, b, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, b, out = ctx.tensors
        grad_a = deepnet.tensor(
            b.data * np.power(a.data, b.data - 1.) * grad.data)
        grad_b = deepnet.tensor(out.data * np.log(a.data) * grad.data)
        return grad_a, grad_b

    @staticmethod
    def jvp(ctx: Context):
        a, b, out = ctx.tensors
        tan_a = b.data * np.power(a.data, b.data - 1.) * a.tangent.data
        tan_b = np.log(a.data) * out.data * b.tangent.data
        tan_out = deepnet.tensor(tan_a + tan_b)
        return tan_out


class Exp(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        out = deepnet.tensor(np.exp(a.data))
        ctx.save(a, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a, out = ctx.tensors
        grad_out = deepnet.tensor(out.data * grad.data)
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a, out = ctx.tensors
        tan_out = deepnet.tensor(out.data * a.tangent.data)
        return tan_out


class Log(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = deepnet.tensor(np.log(a.data))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors[0]
        grad_out = deepnet.tensor(1. / a.data * grad.data)
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        tan_out = deepnet.tensor(1. / a.data * a.tangent.data)
        return tan_out


class Sine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = deepnet.tensor(np.sin(a.data))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors[0]
        grad_a = deepnet.tensor(grad.data * np.cos(a.data))
        return (grad_a,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        tan_out = deepnet.tensor(np.cos(a.data) * a.tangent.data)
        return tan_out


class Cosine(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = deepnet.tensor(np.cos(a.data))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors[0]
        grad_a = deepnet.tensor(grad.data * np.negative(np.sin(a.data)))
        return (grad_a,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        tan_out = deepnet.tensor(a.tangent.data * np.negative(np.sin(a.data)))
        return tan_out


class Sum(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims, keepdims):
        ctx.save(a)
        ctx.a_dim = a.dim()
        ctx.dims = dims
        ctx.keepdims = keepdims
        out = deepnet.tensor(np.sum(a.data, dims, keepdims=keepdims))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a_dim = ctx.a_dim
        dims = ctx.dims
        keepdims = ctx.keepdims
        grad_data = grad.data
        if not keepdims:
            grad_data = np.expand_dims(grad_data, axis=dims)
        grad_out = deepnet.tensor(np.ascontiguousarray(
            np.broadcast_to(grad_data, a_dim)))
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dims = ctx.dims
        keepdims = ctx.keepdims
        tan_out = deepnet.tensor(
            np.sum(a.tangent.data, axis=dims, keepdims=keepdims))
        return tan_out


class Squeeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims: int):
        ctx.save(a)
        ctx.dims = dims
        out = deepnet.tensor(a.data.squeeze(axis=dims))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dims = ctx.dims
        grad_out = deepnet.tensor(np.expand_dims(grad.data, axis=dims))
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dims = ctx.dims
        tan_out = deepnet.tensor(a.tangent.data.squeeze(axis=dims))
        return tan_out


class Unsqueeze(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims: Any):
        ctx.save(a)
        ctx.dims = dims
        out = deepnet.tensor(np.expand_dims(a.data, axis=dims))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dims = ctx.dims
        grad_out = deepnet.tensor(grad.data.squeeze(axis=dims))
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dims = ctx.dims
        tan_out = deepnet.tensor(np.expand_dims(a.tangent.data, axis=dims))
        return tan_out


class View(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim):
        ctx.save(a)
        ctx.a_dim = a.dim()
        ctx.dim = dim
        out = deepnet.tensor(a.data.reshape(dim, order="C"))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a_dim = ctx.a_dim
        grad_out = deepnet.tensor(grad.data.reshape(a_dim, order="C"))
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dim = ctx.dim
        tan_out = deepnet.tensor(a.tangent.data.reshape(dim, order="C"))
        return tan_out


class Reshape(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Any):
        ctx.save(a)
        ctx.a_dim = a.dim()
        ctx.dim = dim
        out = deepnet.tensor(a.data.reshape(dim))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a_dim = ctx.a_dim
        grad_out = deepnet.tensor(grad.data.reshape(a_dim))
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dim = ctx.dim
        tan_out = deepnet.tensor(a.tangent.data.reshape(dim))
        return tan_out


class Tranpose(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim_0: int, dim_1: int):
        out = deepnet.tensor(a.data.swapaxes(dim_0, dim_1))
        ctx.save(a)
        ctx.dim_0 = dim_0
        ctx.dim_1 = dim_1
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dim_0 = ctx.dim_0
        dim_1 = ctx.dim_1
        grad_out = deepnet.tensor(grad.data.swapaxes(dim_0, dim_1))
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dim_0 = ctx.dim_0
        dim_1 = ctx.dim_1
        tan_out = deepnet.tensor(a.tangent.data.swapaxes(dim_0, dim_1))
        return tan_out


class Permute(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dims):
        ctx.save(a)
        ctx.dims = dims
        out = deepnet.tensor(a.data.transpose(dims))
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        dims = np.argsort(ctx.dims)
        grad_out = deepnet.tensor(grad.data.transpose(dims))
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        dims = ctx.dims
        tan_out = deepnet.tensor(a.tangent.data.transpose(dims))
        return tan_out


class Clone(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor):
        ctx.save(a)
        out = deepnet.tensor(a.data.copy())
        return out

    @staticmethod
    def backward(ctx: Any, grad: Tensor):
        return (grad,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        tan_out = deepnet.tensor(a.tangent.data.copy())
        return tan_out


class Slice(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, _slice: slice):
        ctx.save(a)
        ctx.slice = _slice
        ctx.dim = a.dim()
        out = deepnet.tensor(a.data[_slice])
        return out

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        a = ctx.tensors[0]
        _slice = ctx.slice
        mask = np.zeros_like(a.data)
        mask[_slice] = grad.data
        grad_out = deepnet.tensor(mask)
        return (grad_out,)

    @staticmethod
    def jvp(ctx: Context):
        a = ctx.tensors[0]
        _slice = ctx.slice
        tan_out = deepnet.tensor(a.tangent.data[_slice])
        return tan_out
