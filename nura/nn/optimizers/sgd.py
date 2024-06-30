import nura
import nura.nn.utils as utils
from nura.nn.optimizers.optimizer import Optimizer
from nura.nn.parameter import Parameter
from nura.tensors import Tensor
from typing import Optional, Iterator, Tuple


class SGD(Optimizer):

    def __init__(
        self,
        parameters: Iterator[Parameter],
        learnrate: float,
        momentum: float = 0.9,
        nesterov: bool = False,
        decay: Optional[float] = None,
    ) -> None:
        super().__init__(parameters, learnrate, decay)
        self._momentum = momentum
        self._nesterov = nesterov
        self._moments = {}

    @property
    def momentum(self) -> float:
        return self._momentum

    @property
    def nesterov(self) -> bool:
        return self._nesterov

    @property
    def moments(self) -> Iterator[Tuple[Parameter, Tensor]]:
        yield from self._moments.items()

    def step(self) -> None:
        super().step()
        for p in self._parameters:
            if p.grad is None or not p.usegrad:
                continue
            v = self._moments.get(p, nura.zeroslike(p))
            g = sgd(
                parameter=p,
                velocity=v,
                learnrate=self.learnrate,
                momentum=self.momentum,
                nesterov=self.nesterov,
                decay=self.decay,
                graph=False,
            )
            self._moments[p] = g
            self.update(p, g)

    def __repr__(self) -> str:
        learnrate, momentum = self.learnrate, self.momentum
        nesterov, decay = self.nesterov, self.decay
        return f"{self.name()}({learnrate=:.2e} {momentum=} {nesterov=} {decay=})"


def sgd(
    parameter: Parameter,
    velocity: Tensor,
    learnrate: float,
    momentum: float = 0.9,
    nesterov: bool = False,
    decay: Optional[float] = None,
    graph: bool = False,
) -> Tensor:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")

    with nura.setgrad(graph):
        grad = (
            utils.computedecay(parameter, parameter.grad, decay)
            if decay is not None
            else parameter.grad
        )

        if nesterov:
            grad += momentum * velocity
        update = momentum * velocity + learnrate * grad
        return update
