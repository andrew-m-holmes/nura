import nura
import nura.nn.functions as fn
from nura.tensors import Tensor
from nura.utils import where
from typing import Optional, Tuple


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    out = nura.matmul(x, w.T)
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
    out = fn._GELU.apply(z)
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
    mask: Optional[Tensor] = None,
    maskfill=-1e9,
) -> Tuple[Tensor, Tensor]:
    dk = k.dim[-1]
    simscore = nura.matmul(q, k.transpose(-1, -2)) / (dk**0.5)
    if mask is not None:
        simscore = where(mask == True, simscore, maskfill)
    attn = softmax(simscore, -1)
    context = nura.matmul(attn, v)
    return context, attn


def embedding(x: Tensor, w: Tensor, padid: Optional[int] = None):
    if x.dtype not in (nura.int, nura.long):
        raise ValueError(
            f"Expected 'x' to be of type 'int' or 'long' but got '{x.dtype.name()}'"
        )
    return fn._Embedding.apply(x, w, padid)


def crossentropy(z: Tensor, y: Tensor, ignoreid: Optional[int] = None):
    if z.ndim != 2:
        raise ValueError(f"Expected 'z' to be 2D but got '{z.ndim}'D")
    if y.ndim != 1:
        raise ValueError(f"Expected 'y' to be 1D but got '{y.ndim}'D")
    if y.dtype not in (nura.int, nura.long):
        raise ValueError(
            f"Expected 'y' to be of type 'int' or 'long' but got '{y.dtype.name()}'"
        )
    return fn._CrossEntropy.apply(z, y, ignoreid)
