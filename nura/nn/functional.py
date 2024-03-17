import nura.functional as f
import nura.nn.functions as fn
from nura.tensors import Tensor
from nura.utils import atot
from typing import Optional, Tuple


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    out = f.matmul(x, w.transpose())
    if b is not None:
        out = out + b
    return out


def sigmoid(z: Tensor):
    out = 1.0 / (1.0 + f.exp(-z))
    return out


def tanh(z: Tensor):
    e = f.exp(z)
    ne = f.exp(-z)
    out = (e - ne) / (e + ne)
    return out


def relu(z: Tensor):
    z = atot(z)[0]
    out = fn._ReLU.apply(z)
    return out


def relu6(z: Tensor):
    z = atot(z)[0]
    out = fn._ReLU6.apply(z)
    return out


def leakyrelu(z: Tensor, slope=0.01):
    z = atot(z)[0]
    out = fn._LeakyReLU.apply(z, slope)
    return out


def elu(z: Tensor, alpha=1.0):
    z = atot(z)[0]
    out = fn._ELU.apply(z, alpha)
    return out


def gelu(z: Tensor):
    z = atot(z)[0]
    piconst = 0.79788456
    out = 0.5 * z * (1.0 + tanh(piconst * (z + 0.044715 * f.pow(z, 3.0))))
    return out


def softmax(a: Tensor, dim=-1):
    e = f.exp(a)
    out = e / (e.sum(dim, keepdims=False))
    return out


def selfattn(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    scaled = f.matmul(q, k.T) / f.sqrt(k.dim[-1])
    # if mask is not None:
    #     scaled = where(scaled & mask, -1e9, scaled)
    attn = softmax(scaled, dim=-1)
    context = f.matmul(attn, v)
    return context, attn
