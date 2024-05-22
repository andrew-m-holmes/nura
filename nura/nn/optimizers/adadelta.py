import nura
import nura.nn.optimizers.functional as f
from nura.nn.optimizers.optimizer import Optimizer
from nura.nn.parameter import Parameter
from nura.tensors import Tensor
from typing import Optional, Iterator, Tuple


class AdaDelta(Optimizer):

    def __init__(
        self,
        parameters: Iterator[Parameter],
        learnrate: float,
        decay: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learnrate, decay)
        self._deltas = {}
        self._squares = {}
        self._eps = eps

    @property
    def eps(self) -> float:
        return self._eps

    def deltas(self) -> Iterator[Tuple[Parameter, Tensor]]:
        yield from self._deltas.items()

    def squares(self) -> Iterator[Tuple[Parameter, Tensor]]:
        yield from self._squares.items()

    def step(self) -> None:
        super().step()
        for p in self._parameters:
            if p.grad is None or not p.usegrad:
                continue

            s = self._squares.get(p, nura.zeroslike(p))
            d = self._deltas.get(p, nura.zeroslike(p))
