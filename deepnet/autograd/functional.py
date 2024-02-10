import numpy as np
import deepnet
from deepnet.tensors import Tensor
from typing import Tuple, Union, Optional, Callable
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
    *fargs,
    **fkwargs,
) -> Tuple[Tensor, ...]:
    inpt = tupify(inpt)
    assert all(t.gradtensor() for t in inpt)
    assert vec.gradtensor()
    inpt = tuple(t.mutated(usegrad=True) for t in inpt)
    with deepnet.autograd(enabled=True, rev=True):
        out = f(*inpt, *fargs, **fkwargs)
    return grad(inpt, out, vec)


def jvp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    vec: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *fargs,
    **fkwargs,
):
    inpt = tupify(inpt)
    vec = tupify(vec)
    assert all(t.gradtensor() for t in inpt)
    assert all(t.gradtensor() for t in vec)
    inpt = tuple(t.mutated(usegrad=True, grad=g) for t, g in zip(inpt, vec))
    with deepnet.autograd(enabled=True, rev=False):
        out = f(*inpt, *fargs, **fkwargs)
    return out.grad


def jacrev(
    input: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *fargs,
    **fkwargs,
) -> Tuple[Tensor, Tensor]:
    pass

def jacfwd(
    input: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *fargs,
    **fkwargs,
) -> Tuple[Tensor, Tensor]:
    pass

def getjac(inpt, out):
    pass

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
