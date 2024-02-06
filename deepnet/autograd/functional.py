import numpy as np
import deepnet
from collections import deque
from deepnet.tensors import Tensor
from typing import Tuple, Union, List, Optional
from types import FunctionType


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
        if node.accumnode:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = (
                tensor.grad if tensor.grad is not None else deepnet.zeroslike(tensor)
            )
            newgrad = oldgrad.mutated(oldgrad.data + accumgrad.data)
            tensor.mutate(grad=newgrad)
        elif nodes:
            items = [[n, g] for n, g in zip(nodes, node.apply(grad))]
            queue.extend(items)


def grad(
    inpt: Union[Tensor, Tuple[Tensor, ...]], out: Tensor, grad: Tensor
) -> List[Tensor]:
    assert out.backfn is not None
    if grad is None:
        assert out.nelem == 1
        grad = deepnet.oneslike(out)
    inptmap = mapify(inpt)
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
            items = [[n, g] for n, g in zip(nodes, node.apply(grad))]
            queue.extend(items)
    return list(t for t in inptmap.values())


def vjp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    cotan: Tensor,
    f: FunctionType,
    *fargs,
    **fkwargs,
) -> List[Tensor]:
    inpt = tupify(inpt)
    assert all(t.gradtensor() for t in inpt)
    assert (cotan.gradtensor())
    inpt = tuple(t.mutated(usegrad=True) for t in inpt)
    with deepnet.autograd(True):
        out = f(*inpt, *fargs, **fkwargs)
    return grad(inpt, out, cotan)


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
    tans = tuple(t.mutated(usegrad=False) for t in tans)

    with deepnet.autograd(True):
        out = f(*input, *fargs, **fkwargs)

    backfn = out.backfn


def jvptrace(node):
    stack = []
    pass


def mismatch(tensor, grad) -> bool:
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


def mapify(inpt):
    if not isinstance(inpt, tuple):
        inpt = (inpt,)
    return {t: deepnet.zeroslike(t) for t in inpt}


def tupify(inpt) -> Tuple[Tensor, ...]:
    if not isinstance(inpt, tuple):
        return (inpt,)
    return inpt
