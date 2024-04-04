import numpy as np
from nura.autograd.function import Function, Context
from nura.types import dimlike
from nura.tensors import Tensor
from typing import Optional


np._set_promotion_state("weak")


class _Sigmoid(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        arr = 1 / (1 + np.exp(-z.data))
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        return arr * (1 - arr) * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        arr = context["arr"]
        return arr * (1 - arr) * zgrad.data


class _Tanh(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        arr = np.tanh(z.data)
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        return (1 - np.square(arr)) * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        arr = context["arr"]
        return (1 - np.square(arr)) * zgrad.data


class _Softmax(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, dim: int):
        context.save(z)
        exp = np.exp(z.data - z.data.max(axis=dim, keepdims=True))
        arr = exp / exp.sum(axis=dim, keepdims=True)
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        diag = np.diagflat(arr)
        outprod = np.outer(arr, arr)
        jac = diag - outprod
        agg = jac.sum(axis=-1).reshape(arr.shape)
        return agg * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        arr = context["arr"]
        diag = np.diagflat(arr)
        outprod = np.outer(arr, arr)
        jac = diag - outprod
        agg = jac.sum(axis=-1).reshape(arr.shape)
        return agg * zgrad.data


class _ReLU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        return np.maximum(z.data, 0)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        dtype = z.data.dtype
        mask = np.where(z.data > 0, np.array(1, dtype=dtype), np.array(0, dtype=dtype))
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        dtype = z.data.dtype
        mask = np.where(z.data > 0, np.array(1, dtype=dtype), np.array(0, dtype=dtype))
        return mask * zgrad.data


class _ReLU6(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        return np.clip(z.data, 0, 6)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        dtype = z.data.dtype
        mask = np.where(
            (z.data > 0) & (z.data < 6),
            np.array(1, dtype=dtype),
            np.array(0, dtype=dtype),
        )
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        dtype = z.data.dtype
        mask = np.where(
            (z.data > 0) & (z.data < 6),
            np.array(1, dtype=dtype),
            np.array(0, dtype=dtype),
        )
        return mask * zgrad.data


class _LeakyReLU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, slope: float):
        context.save(z)
        context["slope"] = slope
        return np.maximum(z.data * slope, z.data)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        slope = context["slope"]
        dtype = z.data.dtype
        mask = np.where(
            z.data >= 0, np.array(1, dtype=dtype), np.array(slope, dtype=dtype)
        )
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        slope = context["slope"]
        dtype = z.data.dtype
        mask = np.where(
            z.data >= 0, np.array(1, dtype=dtype), np.array(slope, dtype=dtype)
        )
        return mask * zgrad.data


class _ELU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, alpha: float):
        context.save(z)
        context["alpha"] = alpha
        return np.where(z.data > 0, z.data, alpha * (np.exp(z.data) - 1))

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        alpha = context["alpha"]
        dtype = z.data.dtype
        mask = np.where(z.data > 0, np.array(1, dtype=dtype), alpha * np.exp(z.data))
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, zgrad: Tensor):
        z = context.tensors()[0]
        alpha = context["alpha"]
        dtype = z.data.dtype
        mask = np.where(z.data > 0, np.array(1, dtype=dtype), alpha * np.exp(z.data))
        return mask * zgrad.data


class _GELU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor):
        context.save(z)
        PICONST = 0.79788456
        CONST = 0.044715
        context["PICONST"] = PICONST
        context["CONST"] = CONST

        tanh = np.tanh(PICONST * (z.data + CONST * np.power(z.data, 3)))
        inner = tanh + 1.0
        context["tanh"] = tanh
        context["inner"] = inner
        return 0.5 * z.data * inner

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        PICONST = context["PICONST"]
        CONST = context["CONST"]
        tanh = context["tanh"]
        inner = context["inner"]
        dtanh = 1 - tanh**2
        dgelu = 0.5 * (
            inner + z.data * PICONST * dtanh * (1 + 3 * CONST * np.power(z.data, 2))
        )
        return dgelu * grad.data


class _CELU(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, alpha: float):
        context.save(z)
        context["alpha"] = alpha
        arr0 = np.maximum(z.data, 0)
        arr1 = np.minimum(0, alpha * (np.exp(z.data / alpha) - 1))
        return arr0 + arr1

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        alpha = context["alpha"]
        dtype = z.data.dtype
        mask = np.where(z.data >= 0, np.array(1, dtype=dtype), np.exp(z.data / alpha))
        return mask * grad.data


class _Embedding(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, w: Tensor, padid: Optional[int]):
        context.save(w)
        context["xdata"] = x.data
        context["padid"] = padid
        mask = x.data != padid
        mask = np.expand_dims(mask, -1)
        return w.data[x.data] * mask

    @staticmethod
    def backward(context: Context, grad: Tensor):
        w = context.tensors()[0]
        xdata = context["xdata"]
        padid = context["padid"]

        arr = np.zeros_like(w.data)
        mask = xdata != padid
        indices = xdata[mask]
        np.add.at(arr, indices, grad.data[mask])
        return arr


class _CrossEntropy(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, y: Tensor, ignoreid: int):
        context.save(z)
        exp = np.exp(z.data - z.data.max(axis=-1, keepdims=True))
        a = exp / exp.sum(axis=-1, keepdims=True)

        context["a"] = a
        context["ignoreid"] = ignoreid
        context["labels"] = y.data

        mask = y.data != ignoreid
        indices, *_ = mask.nonzero()
        classes = y.data[indices]
        p = a[indices, classes]
        nll = -np.log(p)
        return nll.mean()

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a = context["a"].copy()
        ignoreid = context["ignoreid"]
        labels = context["labels"]

        mask = labels != ignoreid
        indices, *_ = mask.nonzero()
        m = indices.shape[0]
        ignore, *_ = np.invert(mask).nonzero()
        classes = labels[indices]
        a[indices, classes] -= 1
        a[ignore] = 0
        return (a / m) * grad.data


class _Dropout(Function):

    @staticmethod
    def forward(context: Context, z: Tensor, p: float):
        context.save(z)
        mask = np.random.binomial(1, 1 - p, size=z.data.shape)
        context["p"] = p
        context["mask"] = mask
        scale = 1 / (1 - p)
        return (z.data * mask * scale).astype(z.data.dtype)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z = context.tensors()[0]
        p = context["p"]
        mask = context["mask"]
        scale = 1 / (1 - p)
        return (grad.data * mask * scale).astype(z.data.dtype)


class _LayerNorm(Function):

    @staticmethod
    def forward(
        context: Context,
        z: Tensor,
        gamma: Tensor,
        beta: Tensor,
        dim: Optional[dimlike],
        bias: bool,
        eps: float,
    ):
        context.save(z, gamma, beta)
        x = z.data
        mu = x.mean(axis=dim, keepdims=True)
        sigma = np.sqrt(x.var(axis=dim, keepdims=True, ddof=bias) + eps)
        norm = (x - mu) / sigma

        context["dim"] = dim
        context["mu"] = mu
        context["sigma"] = sigma
        context["norm"] = norm
        return gamma.data * norm + beta.data

    @staticmethod
    def backward(context: Context, grad: Tensor):
        z, gamma, beta = context.tensors()
        dim = context["dim"]
        mu = context["mu"]
        sigma = context["sigma"]
        norm = context["norm"]
