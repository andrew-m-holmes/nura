import nura.nn.functional as nnfn
from nura.nn.module import Module
from nura.tensors import Tensor


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return nnfn.relu(x)

    def __repr__(self) -> str:
        return f"{super().__repr__()}()"


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor):
        return nnfn.sigmoid(z)


class Softmax(Module):

    def __init__(self, dim=-1) -> None:
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def forward(self, a: Tensor):
        return nnfn.softmax(a, self.dim)

    def xrepr(self) -> str:
        dim = self.dim
        return f"{super().xrepr()}({dim=})"


class Tanh(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor):
        return nnfn.tanh(z)
