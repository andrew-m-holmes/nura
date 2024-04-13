from nura.tensors import Tensor
from nura.nn import Module, Linear, ScaledDotProductAttention
from nura.types import dtype
from typing import Optional, Tuple, Type


class MultiHeadAttention(Module):

    def __init__(
        self,
        dm: int,
        dk: int,
        dv: int,
        heads: int,
        maskfill=-1e9,
        bias=False,
        dropout: Optional[float] = None,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:

        super().__init__()
        if dm % heads:
            raise ValueError(f"dm ({dm}) must be divisible by heads ({heads})")
        self._dm = dm
        self._dk = dk
        self._dv = dv
        self._heads = heads
        self._maskfill = maskfill

        self._qweight = Linear(dm, heads * dk, bias=bias, dtype=dtype)
        self._kweight = Linear(dm, heads * dk, bias=bias, dtype=dtype)
        self._vweight = Linear(dm, heads * dv, bias=bias, dtype=dtype)
        self._oweight = Linear(heads * dv, dm, bias=bias, dtype=dtype)
        self._attn = ScaledDotProductAttention(maskfill=maskfill, dropout=dropout)

    @property
    def dm(self) -> int:
        return self._dm

    @property
    def dk(self) -> int:
        return self._dk

    @property
    def dv(self) -> int:
        return self._dv

    @property
    def heads(self) -> int:
        return self._heads

    @property
    def qweight(self) -> Linear:
        return self._qweight

    @property
    def kweight(self) -> Linear:
        return self._kweight

    @property
    def vweight(self) -> Linear:
        return self._vweight

    @property
    def oweight(self) -> Linear:
        return self._oweight

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        qlen = q.dim[1]
        klen = k.dim[1]
        q = self._qweight(q).reshape((-1, qlen, self._heads, self._dk)).transpose(1, 2)
        k = self._kweight(k).reshape((-1, klen, self._heads, self._dk)).transpose(1, 2)
        v = self._vweight(v).reshape((-1, klen, self._heads, self._dv)).transpose(1, 2)

        ctx, attn = self._attn(q, k, v, mask=mask)
        ctx = ctx.reshape((-1, qlen, self._heads * self._dv))
        out = self._oweight(ctx)
        return out, attn

    def xrepr(self) -> str:
        dm, dk, dv = self.dm, self.dk, self.dv
        heads = self._heads
        return f"{self.name()}({dm=} {dk=} {dv=} {heads=})"
