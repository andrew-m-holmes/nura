import nura.utils as utils
import nura.functional as f
from nura.tensors import Tensor
from nura.nn.parameter import Parameter
from typing import Iterator, Optional, Tuple


class Optimizer:

    def __init__(
        self,
        parameters: Iterator[Parameter],
        learnrate: float,
        decay: Optional[float] = None,
    ) -> None:
        self._stepnum = 0
        self._parameters = list(parameters)
        self._learnrate = learnrate
        self._decay = decay

    @property
    def learnrate(self) -> float:
        return self._learnrate

    @property
    def decay(self) -> Optional[float]:
        return self._decay

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def update(self, parameter: Tensor, gradstep: Tensor) -> None:
        parameter.detach()
        parameter -= gradstep
        parameter.attach()

    def zerograd(self) -> None:
        for p in self._parameters:
            p.zerograd()

    def step(self) -> None:
        self._stepnum += 1

    def __repr__(self) -> str:
        learnrate, decay = self.learnrate, self.decay
        return f"{self.name()}({learnrate=:.2e} {decay=})"


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
            v = self._moments.get(p, utils.zeroslike(p))
            g = sgd(p, v, self.learnrate, self.momentum, self.nesterov, self.decay)
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
) -> Tensor:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")
    grad = parameter.clone().detached()
    if decay is not None:
        grad += decay * parameter.detached()
    if nesterov:
        grad += momentum * velocity
    update = momentum * velocity + learnrate * grad
    return update


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

    def step(self) -> None:
        super().step()
        for p in self._parameters:
            if p.grad is None or not p.usegrad:
                continue
            v = self._moments.get(p, utils.zeroslike(p))
            g, v_ = rmsprop(p, v, self.learnrate, self.alpha, self.decay, self.eps)
            self._moments[p] = v_
            self.update(p, g)

    def moments(self) -> Iterator[Tuple[Tensor, Tensor]]:
        yield from self._moments.items()

    def __repr__(self) -> str:
        learnrate, alpha, eps, decay = self.learnrate, self.alpha, self.eps, self.decay
        return f"{self.name()}({learnrate=:.2e} {alpha=} {decay=} {eps=:.3e})"


def rmsprop(
    parameter: Parameter,
    velocity: Tensor,
    learnrate: float,
    alpha: float = 0.9,
    decay: Optional[float] = None,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")
    grad = parameter.clone().detached()
    if decay is not None:
        grad += decay * parameter.detached()
    nextvel = alpha * velocity + (1 - alpha) * f.square(grad)
    update = learnrate / f.sqrt(nextvel + eps) * grad
    return update, nextvel


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
        self._moments0 = {}
        self._moments1 = {}

    @property
    def betas(self) -> Tuple[float, float]:
        return self._betas

    @property
    def eps(self) -> float:
        return self._eps

    def __repr__(self) -> str:
        learnrate, betas, eps, decay = self.learnrate, self.betas, self.eps, self.decay
        return f"{self.name()}({learnrate=:.2e} {betas=} {decay=} {eps=})"
