import nura
import nura.nn.functional as f
from nura.tensors import Tensor
from nura.nn.module import Module
from nura.types import dtype, dimlike
from nura.nn.parameter import Parameter, parameter
from typing import Optional, Type, Union


class LayerNorm(Module):

    def __init__(
        self,
        normdim: dimlike,
        unbiased: Union[bool, int] = True,
        eps: float = 1e-5,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:

        super().__init__()
        if dtype is None:
            dtype = nura.float
        self._normdim = normdim
        self._unbiased = unbiased
        self._eps = eps
        self._dtype = dtype
        self._gamma = parameter(nura.randn(normdim), dtype=dtype)
        self._beta = parameter(nura.randn(normdim), dtype=dtype)
        self._dim = tuple(-i for i in range(self.gamma.ndim, 0, -1))

    @property
    def normdim(self) -> dimlike:
        return self._normdim

    @property
    def unbiased(self) -> Union[bool, int]:
        return self._unbiased

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
        return f.layernorm(x, self.gamma, self.beta, self._dim, self.unbiased, self.eps)

    def xrepr(self) -> str:
        normdim, unbiased = self.normdim, self.unbiased
        eps, dtype = self.eps, self.dtype.name()
        return f"{self.name()}({normdim=} {unbiased=} {eps=:.1e} {dtype=})"
