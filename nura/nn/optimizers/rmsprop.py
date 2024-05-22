import nura
import nura.nn.optimizers.functional as f
from nura.nn.optimizers.optimizer import Optimizer
from nura.nn.parameter import Parameter
from nura.tensors import Tensor
from typing import Optional, Iterator, Tuple


class RMSProp(Optimizer):

    def __init__(
        self,
        parameters: Iterator[Parameter],
        learnrate: float,
        alpha: float = 0.9,
        decay: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learnrate, decay)
        self._alpha = alpha
        self._eps = eps
        self._moments = {}

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def eps(self) -> float:
        return self._eps

    def moments(self) -> Iterator[Tuple[Tensor, Tensor]]:
        yield from self._moments.items()

    def step(self) -> None:
        super().step()
        for p in self._parameters:
            if p.grad is None or not p.usegrad:
                continue
            v = self._moments.get(p, nura.zeroslike(p))
            g, v_ = f.rmsprop(
                parameter=p,
                velocity=v,
                learnrate=self.learnrate,
                alpha=self.alpha,
                decay=self.decay,
                eps=self.eps,
                graph=False,
            )
            self._moments[p] = v_
            self.update(p, g)

    def __repr__(self) -> str:
        learnrate, alpha, eps, decay = self.learnrate, self.alpha, self.eps, self.decay
        return f"{self.name()}({learnrate=:.2e} {alpha=} {decay=} {eps=:.3e})"
