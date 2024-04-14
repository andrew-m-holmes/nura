from nura.tensors import Tensor
from nura.nn.parameter import Parameter
from typing import Iterator, Optional, Tuple


class Optimizer:

    def __init__(
        self,
        params: Iterator[Parameter],
        learnrate: float,
        decay: Optional[float] = None,
    ) -> None:
        self._stepnum = 1
        self._params = params
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

    def update(self, a: Tensor, grad: Tensor) -> None:
        a.mutate(usegrad=False)
        a -= self.learnrate * grad
        a.mutate(usegrad=True)

    def autodecay(self, param: Parameter, grad: Tensor) -> Tensor:
        if self.decay is not None:
            return grad + self.decay * param.clone().detach()
        return grad.clone().detach()

    def zerograd(self) -> None:
        for p in self._params:
            p.zerograd()

    def step(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        learnrate, decay = self.learnrate, self.decay
        return f"{self.name()}({learnrate=} {decay=})"


class SGD(Optimizer):

    def __init__(
        self,
        params: Iterator[Parameter],
        learnrate: float,
        momentum: float = 0.9,
        nesterov: bool = False,
        decay: Optional[float] = None,
    ) -> None:
        super().__init__(params, learnrate, decay)
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
        for p in self._params:
            if p.grad is None or not p.usegrad:
                continue
            grad = self.autodecay(p, p.grad)
            vprev = self._moments.get(p, p.clone().detach())
            v = self.momentum * vprev + (1 - self.momentum) * grad
            gradstep = grad + self.momentum * v if self.nesterov else v
            self._moments[p] = gradstep
            self.update(p, gradstep)

    def __repr__(self) -> str:
        learnrate, momentum, decay = self.learnrate, self.momentum, self.decay
        return f"{self.name()}({learnrate=} {momentum=} {decay=})"


class RMSProp(Optimizer):

    def __init__(
        self,
        params: Iterator[Parameter],
        learnrate: float,
        alpha: float = 0.99,
        eps: float = 1e-8,
        decay: Optional[float] = None,
    ) -> None:
        super().__init__(params, learnrate, decay)
        self._alpha = alpha
        self._eps = eps

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def eps(self) -> float:
        return self._eps

    def __repr__(self) -> str:
        learnrate, alpha, eps, decay = self.learnrate, self.alpha, self.eps, self.decay
        return f"{self.name()}({learnrate=} {alpha=} {eps=:.3e} {decay=})"


class Adam(Optimizer):

    def __init__(
        self,
        params: Iterator[Parameter],
        learnrate: float,
        betas: Tuple[float, float] = (0.9, 0.990),
        eps: float = 1e-8,
        decay: Optional[float] = None,
    ) -> None:
        super().__init__(params, learnrate, decay)
        self._betas = betas
        self._eps = eps

    @property
    def betas(self) -> Tuple[float, float]:
        return self._betas

    @property
    def eps(self) -> float:
        return self._eps

    def __repr__(self) -> str:
        learnrate, betas, eps, decay = self.learnrate, self.betas, self.eps, self.decay
        return f"{self.name()}({learnrate=} {betas=} {eps=} {decay=})"
