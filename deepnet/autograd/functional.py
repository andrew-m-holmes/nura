import numpy as np
import deepnet
from deepnet.tensors import Tensor
from typing import List, Tuple, Union, Optional, Callable
from collections import deque


def backward(out: Tensor, grad: Optional[Tensor] = None) -> None:
    assert out.backfn is not None
    if grad is None:
        assert out.nelem == 1
        grad = deepnet.oneslike(out)
    queue = deque()
    queue.append([out.backfn, grad])

    while queue:
        node, grad = queue.popleft()
        nodes = node.children()
        tensor = node.tensor
        if tensor.leaf:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = (
                tensor.grad if tensor.grad is not None else deepnet.zeroslike(tensor)
            )
            newgrad = oldgrad.mutated(oldgrad.data + accumgrad.data)
            tensor.mutate(grad=newgrad)
        elif nodes:
            items = [[n, g] for n, g in zip(nodes, node.applybackward(grad))]
            queue.extend(items)


def grad(
    inpt: Union[Tensor, Tuple[Tensor, ...]], out: Tensor, grad: Optional[Tensor] = None
) -> Tuple[Tensor, ...]:
    assert out.backfn is not None
    if grad is None:
        assert out.nelem == 1
        grad = deepnet.oneslike(out)
    inpt = tupify(inpt)
    vals = tuple(deepnet.zeroslike(t) for t in inpt)
    inptmap = mapify(inpt, vals)
    queue = deque()
    queue.append([out.backfn, grad])

    while queue:
        node, grad = queue.popleft()
        nodes = node.children()
        tensor = node.tensor
        if tensor in inptmap:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = inptmap[tensor]
            oldgrad.mutate(data=oldgrad.data + accumgrad.data)
        if nodes:
            items = [[n, g] for n, g in zip(nodes, node.applybackward(grad))]
            queue.extend(items)
    return tuple(t for t in inptmap.values())


def vjp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    vec: Tensor,
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    inpt = tupify(inpt)
    assert all(t.gradtensor() for t in inpt)
    assert vec.gradtensor()
    inpt = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in inpt)
    vec = vec.mutated(usegrad=False, grad=None, leaf=False)
    with deepnet.autograd(enabled=True, rev=True):
        out = f(*inpt, *args, **kwargs)
    grads = grad(inpt, out, vec)
    return out.mutated(grad=None, usegrad=False, leaf=False), grads


def jvp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    vec: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    inpt = tupify(inpt)
    vec = tupify(vec)
    assert all(t.gradtensor() for t in inpt)
    assert all(t.gradtensor() for t in vec)
    inpt = tuple(t.mutated(usegrad=True, grad=g) for t, g in zip(inpt, vec))
    with deepnet.autograd(enabled=True, rev=False):
        out = f(*inpt, *args, **kwargs)
        assert out.grad is not None
    grad = out.grad
    return out.mutated(grad=None, usegrad=False, leaf=False), grad


def jacrev(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    pass


def jacfwd(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos: int,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    inpt = tupify(
        inpt,
    )
    with deepnet.autograd(enabled=False):
        out = f(*inpt, *args, **kwargs)
    jac = getjac(inpt, out, pos)
    perts = getperts(inpt, pos)
    for col, pert in zip(np.ndindex(inpt[pos].dim), perts):
        jaccol = genjaccol(inpt, f, pert, pos, *args, **kwargs)
        jac[..., *col] = jaccol
    return out, jac

def genjaccol(inpt: Tuple[Tensor, ...], f: Callable[..., Tensor], pert: Tensor, pos: int, *args, **kwargs):
    vec = tuple(deepnet.zeroslike(t) if i != pos else pert for i, t in enumerate(inpt))
    _, jaccol = jvp(inpt, vec, f, *args, **kwargs)
    return jaccol

def getperts(inpt: Tuple[Tensor, ...], pos: int):
    perts = []
    zeros = deepnet.zeros(inpt[pos].dim).to(inpt[pos].dtype)
    for index in np.ndindex(inpt[pos].dim):
        tmp = zeros.copy()
        tmp[index] = 1.0
        perts.append(tmp)
    return perts


def getjac(inpt: Tuple[Tensor, ...], out: Tensor, pos: int) -> Tensor:
    dim = inpt[pos].dim + out.dim
    jac = deepnet.zeros(dim).to(out.dtype)
    return jac


def mismatch(tensor: Tensor, grad) -> bool:
    return tensor.dim != grad.dim and tensor.ndim <= grad.ndim


def sumgrad(tensor: Tensor, grad: Tensor) -> Tensor:
    dims = sumdims(tensor.dim, grad.dim, tensor.ndim, grad.ndim)
    keepdims = tensor.ndim == grad.ndim
    data = np.sum(grad.data, axis=dims, keepdims=keepdims)
    return grad.mutated(data=data)


def sumdims(tdim, gdim, tndim, gndim):
    paddim = np.pad(tdim, (gndim - tndim, 0), constant_values=0)
    mask = paddim != np.array(gdim)
    return tuple(np.where(mask)[0])


def mapify(inpt, vals):
    return {t: v for t, v in zip(inpt, vals)}


def tupify(inpt) -> Tuple[Tensor, ...]:
    if deepnet.istensor(inpt):
        return (inpt,)
    return inpt
