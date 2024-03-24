import nura.types as types
from nura.nn.module import Module
from nura.types import dtype
from nura.tensors import Tensor
from nura.nn import parameter
from nura.utils import randn, onehot
from nura.functional import matmul
from typing import Optional, Type


class Embedding(Module):

    def __init__(
        self, emdim: int, size: int, dtype: Optional[Type[dtype]] = None
    ) -> None:
        super().__init__()
        self._emdim = emdim
        self._size = size
        self._dtype = types.float if dtype is None else dtype
        self._embed = parameter(randn(emdim, size), dtype=dtype)

    @property
    def emdim(self):
        return self._emdim

    @property
    def size(self):
        return self._size

    @property
    def dtype(self) -> Type[dtype]:
        return self._dtype

    @property
    def embed(self) -> Tensor:
        return self._embed

    def to(self, dtype: Type[types.dtype]):
        m = super().to(dtype)
        m._dtype = dtype
        return m

    def forward(self, x: Tensor) -> Tensor:
        x = onehot(x, self.size, dtype=self.dtype)
        return matmul(x, self.embed.T)
