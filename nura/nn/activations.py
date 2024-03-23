import nura.nn.functional as f
from nura.nn.module import Module
from nura.tensors import Tensor
from typing import Optional


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return f.relu(x)


class ReLU6(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return f.relu6(x)


class LeakyReLU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return f.leakyrelu(x)


class ELU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return f.elu(x)


class GELU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return f.gelu(x)


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor):
        return f.sigmoid(z)


class Softmax(Module):

    def __init__(self, dim=-1) -> None:
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def forward(self, a: Tensor):
        return f.softmax(a, self.dim)

    def xrepr(self) -> str:
        dim = self.dim
        return f"{self.name()}({dim=})"


class Tanh(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor):
        return f.tanh(z)


class ScaledDotProductAttention(Module):

    def __init__(self, dim=-1, maskfill=-1e-9) -> None:
        super().__init__()
        self._dim = dim
        self._maskfill = maskfill

    @property
    def dim(self):
        return self._dim

    @property
    def maskfill(self):
        return self._maskfill

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        return f.attention(q, k, v, self.dim, mask, self.maskfill)

    def xrepr(self) -> str:
        dim = self.dim
        maskfill = self.maskfill
        return f"{self.name()}({dim=} {maskfill=:.1e})"
