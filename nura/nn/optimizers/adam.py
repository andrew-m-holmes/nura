import nura
import nura.nn.utils as utils
from nura.nn.optimizers.optimizer import Optimizer
from nura.nn.parameter import Parameter
from nura.tensors import Tensor
from typing import Optional, Iterator, Tuple


class Adam(Optimizer):

    def __init__(
        self,
        parameters: Iterator[Parameter],
        learnrate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        decay: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learnrate, decay)
        self._betas = betas
        self._eps = eps
        self._moments = {}

    @property
    def betas(self) -> Tuple[float, float]:
        return self._betas

    @property
    def eps(self) -> float:
        return self._eps

    def moments(self) -> Iterator[Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        yield from self._moments.items()

    def step(self) -> None:
        super().step()
        for p in self._parameters:
            if p.grad is None or not p.usegrad:
                continue
            vs = self._moments.get(p, (nura.zeroslike(p), nura.zeroslike(p)))
            g, vs = adam(
                parameter=p,
                velocities=vs,
                learnrate=self.learnrate,
                timestep=self.stepnum,
                betas=self.betas,
                decay=self.decay,
                eps=self.eps,
                graph=False,
            )
            self._moments[p] = vs
            self.update(p, g)

    def __repr__(self) -> str:
        learnrate, betas, eps, decay = self.learnrate, self.betas, self.eps, self.decay
        return f"{self.name()}({learnrate=:.2e} {betas=} {decay=} {eps=})"


def adam(
    parameter: Parameter,
    velocities: Tuple[Tensor, Tensor],
    learnrate: float,
    timestep: int,
    betas: Tuple[float, float] = (0.9, 0.99),
    decay: Optional[float] = None,
    eps: float = 1e-8,
    graph: bool = False,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")

    with nura.setgrad(graph):
        grad = (
            utils.computedecay(parameter, parameter.grad, decay)
            if decay is not None
            else parameter.grad
        )

        mt = betas[0] * velocities[0] + (1 - betas[0]) * grad
        vt = betas[1] * velocities[1] + (1 - betas[1]) * nura.square(grad)
        mthat = 1 / (1 - betas[0] ** timestep) * mt
        vthat = 1 / (1 - betas[1] ** timestep) * vt
        update = mthat / nura.sqrt(vthat + eps) * learnrate
        return update, (mt, vt)
