import nura.nn.functional as f
from nura.tensors import Tensor
from nura.nn import Module


class Dropout(Module):

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self._p = p

    @property
    def p(self) -> float:
        return self._p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return f.dropout(x, self.p)
        return x

    def xrepr(self) -> str:
        p = self.p
        return f"{self.name()}(p={p})"
