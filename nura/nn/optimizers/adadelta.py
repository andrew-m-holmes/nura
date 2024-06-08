import nura
import nura.nn.utils as utils
from nura.nn.optimizers.optimizer import Optimizer
from nura.nn.parameter import Parameter
from nura.tensors import Tensor
from typing import Optional, Iterator, Tuple


class AdaDelta(Optimizer):

    def __init__(
        self,
        parameters: Iterator[Parameter],
        gamma: float = 0.9,
        decay: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, 0.0, decay)
        self._gamma = gamma
        self._eps = eps
        self._deltas = {}
        self._squares = {}

    @property
    def gamma(self) -> float:
        return self._gamma

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

            d = self._deltas.get(p, nura.sqrt(nura.zeroslike(p) + self.eps))
            s = self._squares.get(p, nura.zeroslike(p))
            u, d_, s_ = adadelta(
                parameter=p,
                delta=d,
                square=s,
                decay=self.decay,
                eps=self.eps,
                graph=False,
            )

            self._deltas[p] = d_
            self._squares[p] = s_
            self.update(p, u)

    def __repr__(self) -> str:
        gamma, eps, decay = self.gamma, self.eps, self.decay
        return f"{self.name()}({gamma=:.3e} {decay=} {eps=})"


def adadelta(
    parameter: Parameter,
    delta: Tensor,
    square: Tensor,
    gamma: float = 0.9,
    decay: Optional[float] = None,
    eps: float = 1e-8,
    graph: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")

    with nura.setgrad(graph):
        grad = (
            utils.computedecay(parameter, parameter.grad, decay)
            if decay is not None
            else parameter.grad
        )

    emasquare = gamma * square + (1 - gamma) * grad.square()
    nextsquare = nura.sqrt(emasquare + eps)
    emadelta = gamma * delta + (1 - gamma) * delta.square()
    nextdelta = nura.sqrt(emadelta + eps)
    update = (delta / nextsquare) * grad  # 'delta' intentional
    return update, nextdelta, nextsquare
