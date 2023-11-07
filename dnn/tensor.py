import numpy as np


class Tensor:

    def __init__(self, data, use_grad=False) -> None:
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.grad_fn = None
        self.use_grad = use_grad
        self.is_leaf = True

    def __repr__(self) -> str:
        rep = f"({self.data}, "
        rep += f"grad_fn={self.grad_fn})" if self.use_grad \
            else f"use_grad={self.use_grad})"
        return rep

    def backward(self, grad=None):
        if grad is None:
            self.grad = Tensor(np.ones_like(self.data))
        self.grad_fn.backward(self.grad)
