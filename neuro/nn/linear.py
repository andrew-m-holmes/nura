import neuro.utils as utils
from neuro.nn import Module, Parameter
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

        self._weight: Parameter = Parameter(
            utils.rand((outdim, indim), usegrad=True, dtype=dtype)
        )
        self._bias = (
            Parameter(utils.zeros(outdim, usegrad=True, dtype=dtype)) if bias else None
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
        if self._bias is None:
            return linear(inpt, self.weight.tensor)
        return linear(inpt, self.weight.tensor, self._bias.tensor)
