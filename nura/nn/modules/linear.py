import nura.types as types
from nura.nn import parameter
from nura.utils import randn
from nura.nn.module import Module
from nura.tensors import Tensor
from nura.types import dtype
from nura.nn.functional import linear
from typing import Type, Optional


class Linear(Module):

    def __init__(
        self,
        indim: int,
        outdim: int,
        bias=True,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:

        super().__init__()
        if dtype is None:
            dtype = types.float
        self._indim = indim
        self._outdim = outdim
        self._dtype = dtype
        self._weight = parameter(randn((outdim, indim)), True, dtype)
        self._bias = parameter(randn(outdim), True, dtype) if bias else None

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

    @property
    def dtype(self):
        return self._dtype

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

    def to(self, dtype: Type[types.dtype]):
        mod = super().to(dtype)
        mod._dtype = dtype
        return mod

    def xrepr(self) -> str:
        inout = (self.indim, self.outdim)
        bias = True if self.bias is not None else False
        dtype = self.dtype.name()
        return f"{self.name()}({inout=} {bias=} {dtype=})"
