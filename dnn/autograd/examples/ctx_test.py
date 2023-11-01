import torch
from torch.autograd import Function


class MulConstant(Function):
    @staticmethod
    def forward(tensor, constant):
        return tensor * constant

    @staticmethod
    def setup_context(ctx, inputs, output):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor, constant = inputs
        print(ctx, type(ctx), ctx.__class__.__name__, sep="\n\n")
        ctx.constant = constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None


def mul_constant(tensor, c=1):
    return MulConstant.apply(tensor, c)


tensor = torch.ones((5, 1), requires_grad=True)
result = mul_constant(tensor, c=10)
