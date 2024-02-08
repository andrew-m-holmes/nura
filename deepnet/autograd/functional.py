import numpy as np
import deepnet
from deepnet.tensors import Tensor
from typing import Tuple, Union, Optional
from types import FunctionType
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
        nodes = node.nxtnodes()
        tensor = node.tensor
        if tensor.leaf:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = (
                tensor.grad if tensor.grad is not None else deepnet.zeroslike(tensor)
            )
            newgrad = oldgrad.mutated(oldgrad.data + accumgrad.data)
            tensor.mutate(grad=newgrad)
        elif nodes:
            items = [[n, g] for n, g in zip(nodes, node.applyback(grad))]
            queue.extend(items)


def grad(
    inpt: Union[Tensor, Tuple[Tensor, ...]], out: Tensor, grad: Tensor
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
        nodes = node.nxtnodes()
        tensor = node.tensor
        if tensor in inptmap:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = inptmap[tensor]
            oldgrad.mutate(data=oldgrad.data + accumgrad.data)
        if nodes:
            items = [[n, g] for n, g in zip(nodes, node.applyback(grad))]
            queue.extend(items)
    return tuple(t for t in inptmap.values())


def vjp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    cot: Tensor,
    f: FunctionType,
    *fargs,
    **fkwargs,
) -> Tuple[Tensor, ...]:
    inpt = tupify(inpt)
    assert all(t.gradtensor() for t in inpt)
    assert cot.gradtensor()
    inpt = tuple(t.mutated(usegrad=True) for t in inpt)
    with deepnet.autograd(True):
        out = f(*inpt, *fargs, **fkwargs)
    return grad(inpt, out, cot)


def jvp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    tans: Union[Tuple[Tensor, ...], Tensor],
    f: FunctionType,
    *fargs,
    **fkwargs,
):
    inpt = tupify(inpt)
    tans = tupify(tans)
    assert all(t.gradtensor() for t in inpt)
    assert all(t.gradtensor() for t in tans)
    inpt = tuple(t.mutated(usegrad=True) for t in inpt)
    inptmap = mapify(inpt, tans)
    with deepnet.autograd(True, rev=False):
        out = f(*inpt, *fargs, **fkwargs)

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
