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
