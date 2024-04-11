import nura.nn.functional as f
from nura.nn.module import Module
from nura.tensors import Tensor
from typing import Optional, Tuple


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return f.relu(x)


class ReLU6(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return f.relu6(x)


class LeakyReLU(Module):

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def forward(self, x: Tensor) -> Tensor:
        return f.leakyrelu(x, self.alpha)

    def xrepr(self) -> str:
        alpha = self.alpha
        return f"{self.name()}({alpha=})"


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

    def forward(self, x: Tensor) -> Tensor:
        return f.gelu(x)


class CELU(Module):

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def forward(self, x: Tensor) -> Tensor:
        return f.celu(x, self.alpha)

    def xrepr(self) -> str:
        alpha = self.alpha
        return f"{self.name()}({alpha=})"


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor) -> Tensor:
        return f.sigmoid(z)


class Softmax(Module):

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, a: Tensor) -> Tensor:
        return f.softmax(a, self.dim)

    def xrepr(self) -> str:
        dim = self.dim
        return f"{self.name()}({dim=})"


class Tanh(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor) -> Tensor:
        return f.tanh(z)


class ScaledDotProductAttention(Module):

    def __init__(
        self, maskfill: float = -1e-9, dropout: Optional[float] = None
    ) -> None:
        super().__init__()
        self._maskfill = maskfill
        self._dropout = dropout

    @property
    def maskfill(self) -> float:
        return self._maskfill

    @property
    def dropout(self) -> Optional[float]:
        return self._dropout

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return f.attention(q, k, v, mask, self.maskfill, self.dropout)

    def xrepr(self) -> str:
        maskfill, dropout = self.maskfill, self.dropout
        return f"{self.name()}({maskfill=:.1e} {dropout=})"
