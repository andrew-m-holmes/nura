import nura
from nura.nn.module import Module
from nura.tensors import Tensor
from nura.types import dtype
from nura.nn.functional import linear
from typing import Type


class Linear(Module):

    def __init__(
        self,
        indim: int,
        outdim: int,
        bias=True,
        dtype: Type[dtype] = nura.float,
    ) -> None:

        super().__init__()
        self._dtype = dtype
        self._weight = self.param(nura.randn((outdim, indim)))
        self._bias = self.param(nura.randn(outdim)) if bias else None
        self._indim: int = indim
        self._outdim: int = outdim

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

    def xrepr(self) -> str:
        inout = (self.indim, self.outdim)
        bias = True if self.bias is not None else False
        dtype = self.dtype.name()
        return f"{self.__class__.__name__}({inout=} {bias=} {dtype=})"
