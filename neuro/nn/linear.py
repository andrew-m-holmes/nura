import neuro.utils as utils
from neuro.nn.module import Module, Parameter
from neuro.tensors import Tensor
from neuro.types import dtype, float
from neuro.nn.functional import linear
from typing import Type, Optional


class Linear(Module):

    def __init__(
        self, indim: int, outdim: int, bias=True, dtype: Optional[Type[dtype]] = None
    ) -> None:
        super().__init__()

        if dtype is None:
            dtype = float
        self._weight: Parameter = Parameter(utils.randn((outdim, indim)), dtype=dtype)
        self._bias: Optional[Parameter] = (
            Parameter(utils.randn(outdim), dtype=dtype) if bias else None
        )
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

    def forward(self, inpt: Tensor):
        return linear(inpt, self.weight, self.bias)

    def __repr__(self) -> str:
        indim, outdim, bias = self.indim, self.outdim, self.bias is not None
        return f"{super().__repr__()}({indim=} {outdim=} {bias=})"
