import numpy as np
from deepnet import Tensor
from deepnet.autograd.function import Function

# TODO update functions with new context and add type inference
# also don't use tensor operations (that modify state) on forward or backward pass (prevent side effects)


class Add(Function):

    @staticmethod
    def forward(a: Tensor, b: Tensor):
        out = a.data + b.data
        return Tensor(out)

    @staticmethod
    def create_context(context, *tensors):
        context.save_for_backward(*tensors)
        return context

    @staticmethod
    def backward(context, grad):
        grad_a = 1 * grad.data
        grad_b = 1 * grad.data
        return Tensor(grad_a), Tensor(grad_b)


class Sub(Function):

    @staticmethod
    def forward(a: Tensor, b: Tensor):
        out = a.data - b.data
        return Tensor(out)

    @staticmethod
    def create_context(context, *tensors):
        context.save_for_backward(*tensors)
        return context

    @staticmethod
    def backward(context, grad):
        return grad, grad


class Mul(Function):

    @staticmethod
    def forward(a: Tensor, b: Tensor):
        out = a.data * b.data
        return Tensor(out)

    @staticmethod
    def create_context(context, *tensors):
        context.save_for_backward(*tensors)
        return context

    @staticmethod
    def backward(context, grad):
        a, b = context.saved_tensors()
        grad_a = b.data * grad.data
        grad_b = a.data * grad.data
        return Tensor(grad_a), Tensor(grad_b)


class Div(Function):

    @staticmethod
    def forward(a: Tensor, b: Tensor):
        out = a.data / b.data
        return Tensor(out)

    @staticmethod
    def create_context(context, *tensors):
        context.save_for_backward(*tensors)
        return context

    @staticmethod
    def backward(context, grad):
        a, b = context.saved_tensors()
        grad_a = 1 / b.data * grad.data
        grad_b = -1 * a.data / b.data ** 2 * grad.data
        return Tensor(grad_a), Tensor(grad_b)


class Matmul(Function):

    @staticmethod
    def forward(a: Tensor, b: Tensor):
        out = a.data @ b.data
        return Tensor(out)

    @staticmethod
    def backward(context, grad):
        a, b = context.saved_tensors()
        dim_a, dim_b = np.arange(a.ndim()), np.arange(b.ndim())
        dim_a[-2], dim_a[-1] = dim_a[-1], dim_a[-2]
        dim_b[-2], dim_b[-1] = dim_b[-1], dim_b[-2]
        grad_a = grad.data @ b.data.transpose(dim_b)
        grad_b = a.data.transpose(dim_a) @ grad.data
        return Tensor(grad_a), Tensor(grad_b)

    @staticmethod
    def create_context(context, *tensors):
        context.save_for_backward(*tensors)
        return context


class Pow(Function):

    @staticmethod
    def forward(a: Tensor, b: Tensor):
        out = np.power(a.data, b.data)
        return Tensor(out)

    @staticmethod
    def backward(context, grad):
        a, b = context.saved_tensors()
        grad_a = b.data * np.power(a.data, b.data - 1) * grad.data
        grad_b = np.power(a.data, b.data) * np.log(a.data) * grad.data
        return Tensor(grad_a), Tensor(grad_b)

    @staticmethod
    def create_context(context, *args):
        context.save_for_backward(*args)
        return context


class Tranpose(Function):

    @staticmethod
    def forward(tensor: Tensor, dim_0: int, dim_1: int):
        size = np.arange(tensor.ndim())
        size[dim_0], size[dim_1] = size[dim_1], size[dim_0]
        out = tensor.data.transpose(size)
        return Tensor(out)

    @staticmethod
    def backward(context, grad):
        tensor = context.saved_tensors()[0]
        dim_0, dim_1 = context.dim_0, context.dim_1
        size = np.arange(tensor.ndim())
        size[dim_0], size[dim_1] = size[dim_1], size[dim_0]
        grad_data = grad.data.transpose(size)
        return (Tensor(grad_data),)

    @staticmethod
    def create_context(context, *args):
        context.save_for_backward(args[0])
        context.dim_0 = args[1]
        context.dim_1 = args[2]
        return context
