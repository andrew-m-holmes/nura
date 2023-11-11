from deepnet.tensor import Tensor
from deepnet.autograd.function import Function


class Add(Function):

    @staticmethod
    def forward(*tensors: Tensor):
        a, b = tensors
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
    def forward(*tensors: Tensor):
        a, b = tensors
        out = a.data - b.data
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


class Mul(Function):

    @staticmethod
    def forward(*tensors: Tensor):
        a, b = tensors
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
