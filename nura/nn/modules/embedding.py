import nura.types as types
import nura.nn.functional as f
from nura.utils import randn
from nura.nn.module import Module
from nura.types import dtype
from nura.tensors import Tensor
from nura.nn import parameter
from typing import Optional, Type


class Embedding(Module):

    def __init__(
        self,
        emdim: int,
        vocab: int,
        padid: Optional[int] = None,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:
        super().__init__()
        self._emdim = emdim
        self._vocab = vocab
        self._padid = padid
        self._dtype = types.float if dtype is None else dtype
        self._embed = parameter(randn(vocab, emdim), dtype=dtype)

    @property
    def emdim(self):
        return self._emdim

    @property
    def vocab(self):
        return self._vocab

    @property
    def padid(self):
        return self._padid

    @property
    def dtype(self) -> Type[dtype]:
        return self._dtype

    @property
    def embed(self) -> Tensor:
        return self._embed

    def to(self, dtype: Type[types.dtype]):
        mod = super().to(dtype)
        mod._dtype = dtype
        return mod

    def forward(self, x: Tensor) -> Tensor:
        return f.embedding(x, self.embed, self.padid)

    def xrepr(self) -> str:
        emdim, vocab = self.emdim, self.vocab
        padid, dtype = self.padid, self.dtype.name()
        return f"{self.name()}({emdim=} {vocab=} {padid=} {dtype=})"
