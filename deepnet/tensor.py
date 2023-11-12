import numpy as np
import deepnet.nn.functional as f


class Tensor:

    def __init__(self, data, use_grad=False) -> None:
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.grad_fn = None
        self.use_grad = use_grad
        self.is_leaf = True

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        self.grad_fn.apply(grad)

    def __repr__(self) -> str:
        rep = f"({self.data}, "
        rep += f"grad_fn={self.grad_fn})" if self.use_grad \
            else f"use_grad={self.use_grad})"
        return rep

    def __add__(self, other: "Tensor") -> "Tensor":
        return f.add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return f.sub(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return f.mul(self, other)
