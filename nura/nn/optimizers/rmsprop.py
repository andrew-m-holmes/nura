from nura.nn import Parameter
from .optimizer import Optimizer
from typing import Iterator, Optional


class RMSProp(Optimizer):

    def __init__(
        self,
        params: Iterator[Parameter],
        learnrate: float,
        alpha: float = 0.99,
        eps: float = 1e-8,
        decay: Optional[float] = None,
    ) -> None:
        super().__init__(params, learnrate, decay)
        self._alpha = alpha
        self._eps = eps

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def eps(self) -> float:
        return self._eps

    def __repr__(self) -> str:
        learnrate, alpha, eps, decay = self.learnrate, self.alpha, self.eps, self.decay
        return f"{self.name()}({learnrate=} {alpha=} {eps=:.3e} {decay=})"
