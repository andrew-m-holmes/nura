import nura
import nura.nn.optimizers.functional as f
from nura.nn.optimizers.optimizer import Optimizer
from nura.nn.parameter import Parameter
from nura.tensors import Tensor
from typing import Optional, Iterator, Tuple


class AdaGrad(Optimizer):

    def __init__(
        self,
        parameters: Iterator[Parameter],
        learnrate: float,
        decay: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learnrate, decay)
        self._eps = eps
        self._squares = {}

    @property
    def eps(self) -> float:
        return self._eps

    def squares(self) -> Iterator[Tuple[Parameter, Tensor]]:
        yield from self._squares.items()

    def step(self) -> None:
        super().step()
        for p in self._parameters:
            if p.grad is None or not p.usegrad:
                continue

            s = self._squares.get(p, nura.zeroslike(p))
            u, s_ = f.adagrad(
                parameter=p,
                squaregrads=s,
                learnrate=self.learnrate,
                decay=self.decay,
                eps=self.eps,
                graph=False,
            )
            self._squares[p] = s_
            self.update(p, u)

    def __repr__(self) -> str:
        learnrate, eps, decay = self.learnrate, self.eps, self.decay
        return f"{self.name()}({learnrate=:.2e} {decay=} {eps=:.3e})"
