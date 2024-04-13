from nura.nn import Parameter
from .optimizer import Optimizer
from typing import Iterator, Optional


class SGD(Optimizer):

    def __init__(
        self,
        params: Iterator[Parameter],
        learnrate: float,
        momentum: float = 0.9,
        decay: Optional[float] = None,
    ) -> None:
        super().__init__(params, learnrate, decay)
        self._momentum = momentum

    @property
    def momentum(self) -> float:
        return self._momentum

    def __repr__(self) -> str:
        learnrate, momentum, decay = self.learnrate, self.momentum, self.decay
        return f"{self.name()}({learnrate=} {momentum=} {decay=})"
