import nura
from nura.tensors import Tensor
from nura.nn.modules.module import Module
from nura.nn.parameter import Parameter, parameter
from nura.types import dtype
from typing import Optional, Type


class BatchNorm1D(Module):

    def __init__(
        self,
        normdim: int,
        momentum: float = 0.9,
        eps: float = 1e-5,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = nura.float

        self._nordim = normdim
        self._momentum = momentum
        self._eps = eps
        self._dtype = dtype
        self._gamma = parameter(nura.ones(normdim), usegrad=True, dtype=dtype)
        self._beta = parameter(nura.zeros(normdim), usegrad=True, dtype=dtype)
        self._means = []
        self._vars = []

    @property
    def normdim(self) -> int:
        return self._nordim

    @property
    def momentum(self) -> float:
        return self._momentum

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def dtype(self) -> Type[dtype]:
        return self._dtype

    @property
    def gamma(self) -> Parameter:
        return self._gamma

    @property
    def beta(self) -> Parameter:
        return self._beta


def ema(ema: Tensor, stat: Tensor, momentum: float, graph: bool = False) -> Tensor:
    with nura.setgrad(graph):
        return ema * momentum + (1 - momentum) * stat
