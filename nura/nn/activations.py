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

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self._eps = eps

    @property
    def eps(self):
        return self._eps

    def forward(self, z: Tensor):
        return nnfn.sigmoid(z, self.eps)

    def xrepr(self) -> str:
        eps = self.eps
        return f"{super().xrepr()}({eps=})"


class Softmax(Module):

    def __init__(self, pos=-1, eps=1e-6) -> None:
        super().__init__()
        self._pos = pos
        self._eps = eps

    @property
    def pos(self):
        return self._pos

    @property
    def eps(self):
        return self._eps

    def forward(self, a: Tensor):
        return nnfn.softmax(a, self.pos, self.eps)

    def xrepr(self) -> str:
        pos, eps = self.pos, self.eps
        return f"{super().xrepr()}({pos=} {eps=})"


class Tanh(Module):

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self._eps = eps

    @property
    def eps(self):
        return self._eps

    def forward(self, z: Tensor):
        return nnfn.tanh(z, self.eps)

    def xrepr(self) -> str:
        eps = self.eps
        return f"{super().xrepr()}({eps=})"
