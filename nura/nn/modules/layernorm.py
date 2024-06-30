import nura.utils as utils
import nura.types as types
import nura.nn.functional as f
from nura.types import dtype, dimlike
from nura.tensors import Tensor
from nura.nn.modules.module import Module
from nura.nn.parameter import Parameter, parameter
from typing import Optional, Type, Union


class LayerNorm(Module):

    def __init__(
        self,
        normdim: dimlike,
        correction: Union[bool, int] = True,
        eps: float = 1e-5,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:

        super().__init__()
        if dtype is None:
            dtype = types.float
        self._normdim = normdim
        self._correction = correction
        self._eps = eps
        self._dtype = dtype
        self._gamma = parameter(utils.randn(normdim), dtype=dtype)
        self._beta = parameter(utils.randn(normdim), dtype=dtype)
        self._dim = tuple(-i for i in range(self.gamma.ndim, 0, -1))

    @property
    def normdim(self) -> dimlike:
        return self._normdim

    @property
    def correction(self) -> Union[bool, int]:
        return self._correction

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

    def forward(self, x: Tensor) -> Tensor:
        return f.layernorm(x, self.gamma, self.beta, self._dim, self.eps)

    def xrepr(self) -> str:
        normdim, correction = self.normdim, self.correction
        eps, dtype = self.eps, self.dtype.name()
        return f"{self.name()}({normdim=} {correction=} {eps=:.1e} {dtype=})"
