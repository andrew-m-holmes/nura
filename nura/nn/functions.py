import numpy as np
from nura.types import dimlike
from nura.autograd.function import Function, Context
from nura.tensors import Tensor
from typing import Optional, Union


np._set_promotion_state("weak")


class _Sigmoid(Function):

    @staticmethod
    def forward(context: Context, x: Tensor):
        context.save(x)
        arr = 1 / (1 + np.exp(np.negative(x.data)))
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        return arr * (1 - arr) * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        arr = context["arr"]
        return arr * (1 - arr) * grad.data


class _Tanh(Function):

    @staticmethod
    def forward(context: Context, x: Tensor):
        context.save(x)
        arr = np.tanh(x.data)
        context["arr"] = arr
        return arr

    @staticmethod
    def backward(context: Context, grad: Tensor):
        arr = context["arr"]
        return (1 - np.square(arr)) * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        arr = context["arr"]
        return (1 - np.square(arr)) * grad.data


class _Softmax(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, dim: int):
        context.save(x)
        exp = np.exp(x.data - x.data.max(axis=dim, keepdims=True))
        p = exp * (1 / exp.sum(axis=dim, keepdims=True))
        context["p"] = p
        context["dim"] = dim
        return p

    @staticmethod
    def backward(context: Context, grad: Tensor):
        p = context["p"]
        dim = context["dim"]
        outshape = p.shape
        if p.ndim == 1:
            diagonal = np.diagflat(p)
            offdiagonal = np.outer(p, p)
        else:
            p = p.reshape(-1, p.shape[dim])
            diagonal = np.einsum("ij,jk->ijk", p, np.eye(p.shape[dim]))
            offdiagonal = np.einsum("ij,ik->ijk", p, p)
        jac = diagonal - offdiagonal
        return jac.sum(axis=-1).reshape(outshape) * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        p = context["p"]
        dim = context["dim"]
        outshape = p.shape
        if p.ndim == 1:
            diagonal = np.diagflat(p)
            offdiagonal = np.outer(p, p)
        else:
            p = p.reshape(-1, p.shape[dim])
            diagonal = np.einsum("ij,jk->ijk", p, np.eye(p.shape[dim]))
            offdiagonal = np.einsum("ij,ik->ijk", p, p)
        jac = diagonal - offdiagonal
        return jac.sum(axis=-1).reshape(outshape) * grad.data


class _LogSoftmax(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, dim: int):
        context.save(x)
        xmax = x.data.max(axis=dim, keepdims=True)
        logsum = np.log(np.exp(x.data - xmax).sum(axis=dim, keepdims=True))
        nll = np.negative(x.data - xmax - logsum)
        return nll


class _ReLU(Function):

    @staticmethod
    def forward(context: Context, x: Tensor):
        context.save(x)
        return np.maximum(x.data, 0)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x = context.tensors()[0]
        dtype = x.data.dtype
        mask = np.where(x.data > 0, np.array(1, dtype=dtype), np.array(0, dtype=dtype))
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        x = context.tensors()[0]
        dtype = x.data.dtype
        mask = np.where(x.data > 0, np.array(1, dtype=dtype), np.array(0, dtype=dtype))
        return mask * grad.data


class _ReLU6(Function):

    @staticmethod
    def forward(context: Context, x: Tensor):
        context.save(x)
        return np.clip(x.data, 0, 6)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x = context.tensors()[0]
        dtype = x.data.dtype
        mask = np.where(
            (x.data > 0) & (x.data < 6),
            np.array(1, dtype=dtype),
            np.array(0, dtype=dtype),
        )
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        x = context.tensors()[0]
        dtype = x.data.dtype
        mask = np.where(
            (x.data > 0) & (x.data < 6),
            np.array(1, dtype=dtype),
            np.array(0, dtype=dtype),
        )
        return mask * grad.data


class _LeakyReLU(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, alpha: float):
        context.save(x)
        context["alpha"] = alpha
        return np.maximum(x.data * alpha, x.data)

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x = context.tensors()[0]
        alpha = context["alpha"]
        dtype = x.data.dtype
        mask = np.where(
            x.data > 0, np.array(1, dtype=dtype), np.array(alpha, dtype=dtype)
        )
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        x = context.tensors()[0]
        alpha = context["alpha"]
        dtype = x.data.dtype
        mask = np.where(
            x.data > 0, np.array(1, dtype=dtype), np.array(alpha, dtype=dtype)
        )
        return mask * grad.data


class _ELU(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, alpha: float):
        context.save(x)
        context["alpha"] = alpha
        return np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1))

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x = context.tensors()[0]
        alpha = context["alpha"]
        dtype = x.data.dtype
        mask = np.where(x.data > 0, np.array(1, dtype=dtype), alpha * np.exp(x.data))
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        x = context.tensors()[0]
        alpha = context["alpha"]
        dtype = x.data.dtype
        mask = np.where(x.data > 0, np.array(1, dtype=dtype), alpha * np.exp(x.data))
        return mask * grad.data


class _CELU(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, alpha: float):
        context.save(x)
        context["alpha"] = alpha
        arr0 = np.maximum(x.data, 0)
        arr1 = np.minimum(0, alpha * (np.exp(x.data * (1 / alpha)) - 1))
        return arr0 + arr1

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x = context.tensors()[0]
        alpha = context["alpha"]
        dtype = x.data.dtype
        mask = np.where(
            x.data >= 0, np.array(1, dtype=dtype), np.exp(x.data * (1 / alpha))
        )
        return mask * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        x = context.tensors()[0]
        alpha = context["alpha"]
        dtype = x.data.dtype
        mask = np.where(
            x.data >= 0, np.array(1, dtype=dtype), np.exp(x.data * (1 / alpha))
        )
        return mask * grad.data


class _GELU(Function):

    @staticmethod
    def forward(context: Context, x: Tensor):
        context.save(x)
        PICONST = 0.79788456
        CONST = 0.044715
        context["PICONST"] = PICONST
        context["CONST"] = CONST

        tanh = np.tanh(PICONST * (x.data + CONST * x.data * x.data * x.data))
        inner = tanh + 1.0
        context["tanh"] = tanh
        context["inner"] = inner
        return 0.5 * x.data * inner

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x = context.tensors()[0]
        PICONST = context["PICONST"]
        CONST = context["CONST"]
        tanh = context["tanh"]
        inner = context["inner"]
        dtanh = 1 - np.square(tanh)
        dgelu = 0.5 * (
            inner + x.data * PICONST * dtanh * (1 + 3 * CONST * np.square(x.data))
        )
        return dgelu * grad.data

    @staticmethod
    def tangent(context: Context, grad: Tensor):
        x = context.tensors()[0]
        PICONST = context["PICONST"]
        CONST = context["CONST"]
        tanh = context["tanh"]
        inner = context["inner"]
        dtanh = 1 - np.square(tanh)
        dgelu = 0.5 * (
            inner + x.data * PICONST * dtanh * (1 + 3 * CONST * np.square(x.data))
        )
        return dgelu * grad.data


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
    def forward(
        context: Context, x: Tensor, y: Tensor, ignoreid: int, reduction: Optional[str]
    ):
        context.save(x)
        xmax = x.data.max(axis=-1, keepdims=True)
        log = x.data - xmax - np.log(np.exp(x.data - xmax).sum(axis=-1, keepdims=True))
        context["log"] = log
        context["ignoreid"] = ignoreid
        context["labels"] = y.data
        context["reduction"] = reduction

        mask = y.data != ignoreid
        classes = y.data[mask]
        nll = np.negative(log[mask, classes])

        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        if reduction is None:
            return nll

    @staticmethod
    def backward(context: Context, grad: Tensor):
        log = context["log"]
        ignoreid = context["ignoreid"]
        labels = context["labels"]
        reduction = context["reduction"]

        a = np.exp(log)
        mask = labels != ignoreid
        ignore = np.invert(mask)
        classes = labels[mask]
        a[mask, classes] -= 1
        a[ignore] = 0

        if reduction == "mean":
            return (1 / labels.size) * a * grad.data
        if reduction == "sum" or reduction is None:
            return a * grad.data


class _BinaryCrossEntropy(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, y: Tensor, reduction: Optional[str]):
        context.save(a, y)
        nll = np.negative(y.data * np.log(a.data) + (1 - y.data) * np.log(1 - a.data))
        context["reduction"] = reduction

        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        if reduction is None:
            return nll

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, y = context.tensors()
        reduction = context["reduction"]
        arr = np.negative(y.data) * (1 / a.data) + (1 - y.data) * (1 / (1 - a.data))

        if reduction == "mean":
            return (1 / y.data.size) * arr * grad.data
        if reduction == "sum" or reduction is None:
            return arr * grad.data


class _MSE(Function):

    @staticmethod
    def forward(context: Context, a: Tensor, y: Tensor, reduction: Optional[str]):
        context.save(a, y)
        context["reduction"] = reduction
        mse = 0.5 * np.square(a.data - y.data)
        if reduction == "mean":
            return mse.mean()
        if reduction == "sum":
            return mse.sum()
        if reduction is None:
            return mse

    @staticmethod
    def backward(context: Context, grad: Tensor):
        a, y = context.tensors()
        reduction = context["reduction"]
        if reduction == "mean":
            return (1 / y.data.size) * (a.data - y.data) * grad.data
        if reduction == "sum" or reduction is None:
            return (a.data - y.data) * grad.data


class _Dropout(Function):

    @staticmethod
    def forward(context: Context, x: Tensor, p: float):
        context.save(x)
        mask = np.random.binomial(1, 1 - p, size=x.data.shape).astype(x.data.dtype)
        context["p"] = p
        context["mask"] = mask
        scale = 1 / (1 - p) if p < 1 else np.inf
        return x.data * mask * scale

    @staticmethod
    def backward(context: Context, grad: Tensor):
        p = context["p"]
        mask = context["mask"]
        scale = 1 / (1 - p) if p < 1 else np.inf
        return grad.data * mask * scale


class _LayerNorm(Function):

    @staticmethod
    def forward(
        context: Context,
        x: Tensor,
        gamma: Tensor,
        beta: Tensor,
        dim: Optional[dimlike],
        correction: Union[bool, int],
        eps: float,
    ):
        context.save(x, gamma, beta)
        mu = x.data.mean(axis=dim, keepdims=True)
        var = x.data.var(axis=dim, keepdims=True, ddof=correction)
        istd = 1 / np.sqrt(var + eps)
        norm = (x.data - mu) * istd

        context["dim"] = dim
        context["mu"] = mu
        context["istd"] = istd
        context["norm"] = norm
        context["correction"] = correction
        context["eps"] = eps
        return gamma.data * norm + beta.data

    @staticmethod
    def backward(context: Context, grad: Tensor):
        x, gamma, beta = context.tensors()
        dim = context["dim"]
        mu = context["mu"]
        istd = context["istd"]
        norm = context["norm"]
        correction = context["correction"]

        n = (
            x.data.shape[dim]
            if isinstance(dim, int)
            else np.prod([x.data[d] for d in dim])
        )

        dgamma = grad.data * norm
        dbeta = grad.data.copy()
        dnorm = grad.data * gamma.data

        xdiffmu = x.data - mu
        dvar = np.sum(
            dnorm * xdiffmu * -0.5 * np.power(istd, 1.5),
            axis=dim,
            keepdims=True,
        )

        scale = 2 / (n - correction)
        dmu = np.sum(np.negative(dnorm) * istd, axis=dim, keepdims=True)
        dmu += np.negative(scale) * dvar * np.sum(xdiffmu, axis=dim, keepdims=True)

        dx = dnorm * istd
        dx += scale * dvar * xdiffmu
        dx += (1 / n) * dmu
        return dx, dgamma, dbeta
