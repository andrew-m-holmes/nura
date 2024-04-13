from typing import Iterator, Optional
from nura.nn import Parameter


class Optimizer:

    def __init__(
        self,
        params: Iterator[Parameter],
        learnrate: float,
        decay: Optional[float] = None,
    ) -> None:
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
        decay: Optional[float] = None,
    ) -> None:
        super().__init__(params, learnrate, decay)
        self._momentum = momentum

    @property
    def momentum(self) -> float:
        return self._momentum

    def step(self) -> None:
        pass

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
