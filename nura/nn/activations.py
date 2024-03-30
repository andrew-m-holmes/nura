import nura.nn.functional as f
from nura.nn.module import Module
from nura.tensors import Tensor
from typing import Optional, Tuple


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

    def __init__(self, slope: float = 0.1) -> None:
        super().__init__()
        self._slope = slope

    @property
    def slope(self):
        return self._slope

    def forward(self, x: Tensor):
        return f.leakyrelu(x, self.slope)

    def xrepr(self) -> str:
        slope = self.slope
        return f"{self.name()}({slope=})"


class ELU(Module):

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self._alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    def forward(self, x: Tensor):
        return f.elu(x, self.alpha)

    def xrepr(self) -> str:
        alpha = self.alpha
        return f"{self.name()}({alpha=})"


class GELU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return f.gelu(x)


class CELU(Module):

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self._alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    def forward(self, x: Tensor):
        return f.celu(x, self.alpha)

    def xrepr(self) -> str:
        alpha = self.alpha
        return f"{self.name()}({alpha=})"


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

    def __init__(self, maskfill=-1e-9) -> None:
        super().__init__()
        self._maskfill = maskfill

    @property
    def maskfill(self):
        return self._maskfill

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return f.attention(q, k, v, mask, self.maskfill)

    def xrepr(self) -> str:
        maskfill = self.maskfill
        return f"{self.name()}({maskfill=:.1e})"
