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

    def __init__(self, dim=-1, eps=1e-6) -> None:
        super().__init__()
        self._dim = dim
        self._eps = eps

    @property
    def dim(self):
        return self._dim

    @property
    def eps(self):
        return self._eps

    def forward(self, a: Tensor):
        return nnfn.softmax(a, self.dim, self.eps)

    def xrepr(self) -> str:
        dim, eps = self.dim, self.eps
        return f"{super().xrepr()}({dim=} {eps=})"


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
