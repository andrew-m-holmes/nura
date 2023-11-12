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


def add(a: Tensor, b: Tensor) -> Tensor:
    return Add.apply(a, b)


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


def sub(a: Tensor, b: Tensor) -> Tensor:
    return Sub.apply(a, b)


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


def mul(a: Tensor, b: Tensor):
    return Mul.apply(a, b)
