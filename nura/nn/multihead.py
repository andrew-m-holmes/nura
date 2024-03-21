from nura.nn import Module
from nura.nn import parameter
from nura.tensors import Tensor
from nura.nn.linear import Linear
from nura.nn.activations import SelfAttention
from nura.utils import randn


class MultiHeadAttention(Module):

    def __init__(self, emdim, hdim, dim=-1, maskfill=-1e9, bias=False) -> None:
        super().__init__()
        self._emdim = emdim
        self._hdim = hdim
        self._dim = dim
        self._maskfill = maskfill
        self._wq = Linear(emdim, hdim, bias=bias)
        self._wk = Linear(emdim, hdim, bias=bias)
        self._wv = Linear(emdim, hdim, bias=bias)
        self._wo = Linear(hdim, emdim, bias=bias)
        self._attn = SelfAttention(dim=dim, maskfill=maskfill)

    def forward(self, q, k, v, mask=None):
        pass
