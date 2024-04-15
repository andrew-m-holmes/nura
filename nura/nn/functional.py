import nura
import nura.nn.functions as fn
import nura.utils as utils
from nura.types import dimlike
from nura.tensors import Tensor
from typing import Optional, Tuple


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    out = nura.matmul(x, w.T)
    if b is not None:
        out = out + b
    return out


def sigmoid(x: Tensor) -> Tensor:
    out = fn._Sigmoid.apply(x)
    return out


def tanh(x: Tensor) -> Tensor:
    out = fn._Tanh.apply(x)
    return out


def relu(x: Tensor) -> Tensor:
    out = fn._ReLU.apply(x)
    return out


def relu6(x: Tensor) -> Tensor:
    out = fn._ReLU6.apply(x)
    return out


def leakyrelu(x: Tensor, alpha: float = 0.01) -> Tensor:
    out = fn._LeakyReLU.apply(x, alpha)
    return out


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    out = fn._ELU.apply(x, alpha)
    return out


def gelu(x: Tensor) -> Tensor:
    out = fn._GELU.apply(x)
    return out


def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    out = fn._CELU.apply(x, alpha)
    return out


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    out = fn._Softmax.apply(x, dim)
    return out


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    maskfill: float = -1e9,
    drop: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    norm = 1 / (k.dim[-1] ** 0.5)
    simscore = nura.matmul(q, k.transpose(-1, -2)) * norm
    if mask is not None:
        simscore = utils.where(mask == True, simscore, maskfill)
    attn = softmax(simscore, -1)
    if drop is not None:
        attn = dropout(attn, drop)
    context = nura.matmul(attn, v)
    return context, attn


def embedding(x: Tensor, w: Tensor, padid: Optional[int] = None) -> Tensor:
    return fn._Embedding.apply(x, w, padid)


def binarycrossentropy(
    x: Tensor, y: Tensor, reduction: Optional[str] = "mean"
) -> Tensor:
    if y.ndim > 2:
        raise ValueError("'y' cannot be more than 2D")
    if x.ndim != y.ndim:
        raise ValueError("'x' must have the same rank as 'y'")
    return fn._BinaryCrossEntropy.apply(x, y, reduction)


def crossentropy(
    x: Tensor,
    y: Tensor,
    ignoreid: Optional[int] = None,
    reduction: Optional[str] = "mean",
) -> Tensor:
    if x.ndim != 2:
        raise ValueError("'x' must be 2D")
    if y.ndim != 1:
        raise ValueError("'y' must be 1D")
    return fn._CrossEntropy.apply(x, y, ignoreid, reduction)


def mse(a: Tensor, y: Tensor, reduction: Optional[str] = "mean") -> Tensor:
    if a.ndim != y.ndim:
        raise ValueError("'x' must have the same dimensions as 'y'")
    return fn._MSE.apply(a, y, reduction)


def rmse(a: Tensor, y: Tensor, reduction: Optional[str] = "mean") -> Tensor:
    if a.ndim != y.ndim:
        raise ValueError("'x' must have the same dimensions as 'y'")
    return fn._RMSE.apply(a, y, reduction)


def dropout(x: Tensor, p: float = 0.5) -> Tensor:
    if p < 0:
        raise ValueError("'p' cannot be less than zero")
    return fn._Dropout.apply(x, p)


def layernorm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    dim: dimlike = -1,
    correction: int = 1,
    eps: float = 1e-5,
) -> Tensor:
    return fn._LayerNorm.apply(x, gamma, beta, dim, correction, eps)
