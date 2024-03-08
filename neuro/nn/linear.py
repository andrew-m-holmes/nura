import neuro
from neuro.nn.module import Module
from neuro.nn.parameter import Parameter
from neuro.tensors import Tensor
from neuro.types import dtype
from neuro.nn.functional import linear
from typing import Type, Optional


class Linear(Module):

    def __init__(
        self,
        indim: int,
        outdim: int,
        bias=True,
        dtype: Optional[Type[dtype]] = neuro.float,
    ) -> None:

        super().__init__()
        self._weight = self.param(neuro.randn((outdim, indim)), dtype)
        self._bias = self.param(neuro.randn(outdim), dtype) if bias else None
        self._indim: int = indim
        self._outdim: int = outdim
        self._dtype = dtype

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    @property
    def indim(self):
        return self._indim

    @property
    def outdim(self):
        return self._outdim

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)
