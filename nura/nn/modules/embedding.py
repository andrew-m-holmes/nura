import nura.nn.functional as f
import nura.utils as utils
import nura.types as types
from nura.types import dtype
from nura.nn.modules.module import Module
from nura.nn.parameter import Parameter, parameter
from nura.tensors import Tensor
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
        self._weight = parameter(utils.randn(vocab, emdim), dtype=dtype)

    @property
    def emdim(self) -> int:
        return self._emdim

    @property
    def vocab(self) -> int:
        return self._vocab

    @property
    def padid(self) -> Optional[int]:
        return self._padid

    @property
    def dtype(self) -> Type[dtype]:
        return self._dtype

    @property
    def weight(self) -> Parameter:
        return self._weight

    def to(self, dtype: Type[types.dtype]) -> Module:
        mod = super().to(dtype)
        mod._dtype = dtype
        return mod

    def forward(self, x: Tensor) -> Tensor:
        return f.embedding(x, self.weight, self.padid)

    def xrepr(self) -> str:
        emdim, vocab = self.emdim, self.vocab
        padid, dtype = self.padid, self.dtype.name()
        return f"{self.name()}({emdim=} {vocab=} {padid=} {dtype=})"
