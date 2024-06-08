import nura
import nura.nn.utils as utils
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
            u, s_ = adagrad(
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


def adagrad(
    parameter: Parameter,
    squaregrads: Tensor,
    learnrate: float,
    decay: Optional[float] = None,
    eps: float = 1e-8,
    graph: bool = False,
) -> Tuple[Tensor, Tensor]:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")

    with nura.setgrad(graph):
        grad = (
            utils.computedecay(parameter, parameter.grad, decay)
            if decay is not None
            else parameter.grad
        )

        squaregrad = nura.square(grad)
        squaregrads = squaregrads + squaregrad
        update = learnrate / nura.sqrt(squaregrads + eps) * grad
        return update, squaregrads
