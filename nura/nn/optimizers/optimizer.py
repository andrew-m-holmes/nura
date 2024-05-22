import nura
import nura.nn.optimizers.functional as f
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
        self._parameters = tuple(parameters)
        self._learnrate = learnrate
        self._decay = decay

    @property
    def learnrate(self) -> float:
        return self._learnrate

    @property
    def decay(self) -> Optional[float]:
        return self._decay

    @property
    def stepnum(self) -> int:
        return self._stepnum

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def update(self, parameter: Tensor, gradstep: Tensor) -> None:
        with nura.nograd():
            parameter -= gradstep

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
            v = self._moments.get(p, nura.zeroslike(p))
            g = f.sgd(
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
            g, vs = f.adam(
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
