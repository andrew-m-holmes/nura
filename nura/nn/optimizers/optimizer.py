import nura
from nura.tensors import Tensor
from nura.nn.parameter import Parameter
from typing import Iterator, Optional


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
