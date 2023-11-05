import numpy as np
import torch


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
            grad = self.__init__(np.ones_like(self.data))
        print(self.grad_fn)
        self.grad_fn.backward(grad)


if __name__ == "__main__":
    a = [[1, 2, 3,], [4, 5, 6], [7, 8, 9]]
    tensor = Tensor(a)
    print(tensor)
