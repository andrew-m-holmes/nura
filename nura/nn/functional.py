import nura
import nura.nn.functions as fn
import nura.utils as utils
from nura.tensors import Tensor
from nura.types import dimlike
from typing import Optional, Tuple, Union


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None):
    out = nura.matmul(x, w.T)
    if b is not None:
        out = out + b
    return out


def sigmoid(x: Tensor):
    out = fn._Sigmoid.apply(x)
    return out


def tanh(x: Tensor):
    out = fn._Tanh.apply(x)
    return out


def relu(x: Tensor):
    out = fn._ReLU.apply(x)
    return out


def relu6(x: Tensor):
    out = fn._ReLU6.apply(x)
    return out


def leakyrelu(x: Tensor, slope=0.01):
    out = fn._LeakyReLU.apply(x, slope)
    return out


def elu(x: Tensor, alpha=1.0):
    out = fn._ELU.apply(x, alpha)
    return out


def gelu(x: Tensor):
    out = fn._GELU.apply(x)
    return out


def celu(x: Tensor, alpha=1.0):
    out = fn._CELU.apply(x, alpha)
    return out


def softmax(x: Tensor, dim=-1):
    out = fn._Softmax.apply(x, dim)
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
        simscore = utils.where(mask == True, simscore, maskfill)
    attn = softmax(simscore, -1)
    context = nura.matmul(attn, v)
    return context, attn


def embedding(x: Tensor, w: Tensor, padid: Optional[int] = None):
    if x.dtype not in (nura.int, nura.long):
        raise RuntimeError(
            f"Expected 'x' to be of type 'int' or 'long' but got '{x.dtype.name()}'"
        )
    return fn._Embedding.apply(x, w, padid)


def binarycrossentropy(x: Tensor, y: Tensor):
    if x.ndim != 2:
        raise RuntimeError(f"Expected 'a' to be 2D but got '{x.ndim}'D")
    if y.ndim != 1:
        raise RuntimeError(f"Expected 'y' to be 1D but got '{y.ndim}'D")
    if x.dtype not in (nura.half, nura.float, nura.double):
        raise ValueError(
            f"Expected 'a' to be of type 'half', 'float', or 'double' but got '{x.dtype.name()}'"
        )
    if y.dtype not in (nura.half, nura.float, nura.double):
        raise ValueError(
            f"Expected 'y' to be of type 'half', 'float', or 'double' but got '{y.dtype.name()}'"
        )
    return fn._BinaryCrossEntropy.apply(x, y)


def crossentropy(x: Tensor, y: Tensor, ignoreid: Optional[int] = None):
    if x.ndim != 2:
        raise RuntimeError(f"Expected 'x' to be 2D but got '{x.ndim}'D")
    if y.ndim != 1:
        raise RuntimeError(f"Expected 'y' to be 1D but got '{y.ndim}'D")
    if y.dtype not in (nura.int, nura.long):
        raise RuntimeError(
            f"Expected 'y' to be of type 'int' or 'long' but got '{y.dtype.name()}'"
        )
    return fn._CrossEntropy.apply(x, y, ignoreid)


def dropout(x: Tensor, p: float = 0.5):
    return fn._Dropout.apply(x, p)


def layernorm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    dim: dimlike = -1,
    unbiased: Union[bool, int] = True,
    eps: float = 1e-5,
):
    expecteddim = (
        (x.dim[dim],) if isinstance(dim, int) else tuple(x.dim[d] for d in dim)
    )
    if gamma.dim != expecteddim:
        raise ValueError(
            f"Expected 'gamma' to be of dimensions {expecteddim} but got {gamma.dim}"
        )
    if beta.dim != expecteddim:
        raise ValueError(
            f"Expected 'beta' to be of dimensions {expecteddim} but got {beta.dim}"
        )
    return fn._LayerNorm.apply(x, gamma, beta, dim, unbiased, eps)
