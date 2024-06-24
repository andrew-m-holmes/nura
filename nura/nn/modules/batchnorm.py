import nura
import nura.nn.functional as f
from nura.tensors import Tensor
from nura.nn.modules.module import Module
from nura.nn.parameter import Parameter, parameter
from nura.types import dtype
from typing import Optional, Type


class BatchNorm(Module):

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
        self._mean = None
        self._var = None

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

    def forward(self, x: Tensor) -> Tensor:
        if x.dim[-1] != self.normdim:
            raise ValueError(
                f"Expected feature dimension to be {self.normdim} but received {x.dim[-1]}"
            )
        if self.training:
            self._mean = nura.zeros(self.normdim).to(self.dtype)
            self._var = nura.zeros(self.normdim).to(self.dtype)
            dim = tuple(range(x.ndim))[:-1]
            mean = x.clone().detach().mean(dim=dim, keepdims=True)
            var = x.clone().detach().var(dim=dim, keepdims=True)
            varunbiased = x.clone().detach().var(correction=1, dim=dim, keepdims=True)
            self._mean = self.momentum * self._mean + (1 - self.momentum) * mean
            self._var = self.momentum * self._var + (1 - self.momentum) * varunbiased
            return f.batchnorm(x, self.gamma, self.beta, mean, var, self.eps)
        return f.batchnorm(x, self.gamma, self.beta, self._mean, self._var, self.eps)

    def xrepr(self) -> str:
        nordim, momentum, eps, dtype = (
            self.normdim,
            self.momentum,
            self.eps,
            self.dtype.name(),
        )
        return f"{self.name()}({nordim=} {momentum=} {eps=:.2e} {dtype=})"
