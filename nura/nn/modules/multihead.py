from nura.nn import Module
from nura.tensors import Tensor
from nura.nn.modules.linear import Linear
from nura.nn.activations import ScaledDotProductAttention
from nura.types import dtype
from typing import Optional, Tuple, Type


class MultiHeadAttention(Module):

    def __init__(
        self,
        dm: int,
        dk: int,
        dv: int,
        heads: int,
        dim=-1,
        maskfill=-1e9,
        bias=False,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:

        super().__init__()
        if dm % heads:
            raise ValueError(f"dm ({dm}) must be divisible by heads ({heads})")
        self._dm = dm
        self._dk = dk
        self._dv = dv
        self._heads = heads
        self._dim = dim
        self._maskfill = maskfill

        self._wq = Linear(dm, heads * dk, bias=bias, dtype=dtype)
        self._wk = Linear(dm, heads * dk, bias=bias, dtype=dtype)
        self._wv = Linear(dm, heads * dv, bias=bias, dtype=dtype)
        self._wo = Linear(heads * dv, dm, bias=bias, dtype=dtype)
        self._attn = ScaledDotProductAttention(dim=dim, maskfill=maskfill)

        # (m, l, h, e) -> (m, h, l, e)
        # (m, h, l, e) @ (m, h, e, l) -> (m, h, l, l)
        # (m, h, l, l) @ (m, h, l, v) -> (m, h, l, v)

    def forward(self, q, k, v, mask=None) -> Tuple[Tensor, Tensor]:
        seqlen = k.dim[1]
        q = self._wq(q).reshape(-1, seqlen, self._heads, self._dk).transpose(1, 2)
        k = self._wk(k).reshape(-1, seqlen, self._heads, self._dk).transpose(1, 2)
        v = self._wv(v).reshape(-1, seqlen, self._heads, self._dv).transpose(1, 2)

        ctx, attn = self._attn(q, k, v, mask=mask)
        ctx = ctx.reshape(-1, seqlen, self._heads * self._dv)
        out = self._wo(ctx)
        return out, attn
