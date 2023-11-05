from function import Function
from dnn.tensor import Tensor
from typing import Tuple

if __name__ == "__main__":

    a = Tensor([2], use_grad=True)
    b = Tensor([3], use_grad=True)

    class Mul(Function):

        @staticmethod
        def forward(ctx, a: Tensor, b: Tensor):
            ctx.save_for_backward(a, b)
            return Tensor(a.data * b.data)

        @staticmethod
        def backward(ctx, grad: Tensor) -> Tuple[Tensor, Tensor]:
            a, b = ctx.saved_tensors
            grad_a = Tensor(grad.data * b.data)
            grad_b = Tensor(grad.data * a.data)
            return grad_a, grad_b

    c = Mul.apply(a, b)
    print(c.grad_fn)
    c.backward()
