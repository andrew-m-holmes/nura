import nura.types as types
import nura.nn.functional as f
import nura.utils as utils
from nura.nn.modules.module import Module
from nura.nn.parameter import Parameter, parameter
from nura.nn.utils import he
from nura.tensors import Tensor
from nura.types import dtype
from typing import Type, Optional, Callable


class Linear(Module):

    def __init__(
        self,
        inputdim: int,
        outputdim: int,
        bias: bool = True,
        dtype: Optional[Type[dtype]] = None,
        init: Optional[Callable[..., Tensor]] = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = types.float
        if init is None:
            init = he

        self._inputdim = inputdim
        self._outputdim = outputdim
        self._dtype = dtype
        self._init = init
        self._weight = parameter(init(inputdim, outputdim), dtype=dtype)
        self._bias = parameter(utils.randn(outputdim), dtype=dtype) if bias else None

    @property
    def weight(self) -> Parameter:
        return self._weight

    @property
    def bias(self) -> Optional[Parameter]:
        return self._bias

    @property
    def inputdim(self) -> int:
        return self._inputdim

    @property
    def outputdim(self) -> int:
        return self._outputdim

    @property
    def dtype(self) -> Type[dtype]:
        return self._dtype

    def forward(self, x: Tensor) -> Tensor:
        return f.linear(x, self.weight, self.bias)

    def to(self, dtype: Type[types.dtype]) -> Module:
        mod = super().to(dtype)
        mod._dtype = dtype
        return mod

    def xrepr(self) -> str:
        inputdim, outputdim = self.inputdim, self.outputdim
        bias = True if self.bias is not None else False
        dtype = self.dtype.name()
        return f"{self.name()}({inputdim=} {outputdim=} {bias=} {dtype=})"
