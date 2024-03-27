import nura
import nura.functional as f
import nura.nn.functions as fn
from nura.tensors import Tensor
from nura.utils import where
from typing import Optional, Tuple


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    out = nura.matmul(x, w.transpose())
    if b is not None:
        out = out + b
    return out


def sigmoid(z: Tensor):
    out = fn._Sigmoid.apply(z)
    return out


def tanh(z: Tensor):
    out = fn._Tanh.apply(z)
    return out


def relu(z: Tensor):
    out = fn._ReLU.apply(z)
    return out


def relu6(z: Tensor):
    out = fn._ReLU6.apply(z)
    return out


def leakyrelu(z: Tensor, slope=0.01):
    out = fn._LeakyReLU.apply(z, slope)
    return out


def elu(z: Tensor, alpha=1.0):
    out = fn._ELU.apply(z, alpha)
    return out


def gelu(z: Tensor):
    piconst = 0.79788456
    const = 0.044715
    inner = piconst * (z + const * f.pow(z, 3.0))
    out = 0.5 * z * (1 + tanh(inner))
    return out


def celu(z: Tensor, alpha=1.0):
    out = fn._CELU.apply(z, alpha)
    return out


def softmax(a: Tensor, dim=-1):
    out = fn._Softmax.apply(a, dim)
    return out


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dim=-1,
    mask: Optional[Tensor] = None,
    maskfill=-1e9,
) -> Tuple[Tensor, Tensor]:
    dk = k.dim[dim]
    simscore = nura.matmul(q, k.transpose(-1, -2)) / (dk**0.5)
    if mask is not None:
        simscore = where(mask == True, simscore, maskfill)
    attn = softmax(simscore, dim)
    context = nura.matmul(attn, v)
    return context, attn


def embedding(x: Tensor, w: Tensor, padid: Optional[int] = None):
    if x.dtype not in (nura.int, nura.long):
        raise ValueError(
            f"Expected 'x' to be of type 'int' or 'long' but got '{x.dtype.name()}'"
        )
    return fn._Embedding.apply(x, w, padid)
