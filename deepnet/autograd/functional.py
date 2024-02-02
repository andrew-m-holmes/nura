import numpy as np
import deepnet
from collections import deque
from deepnet.tensors import Tensor
from typing import Tuple, Union, List, Optional


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
        if tensor.leaf and tensor.mutable:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = tensor.grad if tensor.grad else deepnet.oneslike(tensor)
            newgrad = oldgrad.mutated(oldgrad.data + accumgrad.data)
            tensor.mutate(grad=newgrad)
        if nodes:
            items = [[n, g] for n, g in zip(nodes, node.apply(grad))]
            queue.extend(items)


def grad(
    inpt: Union[Tensor, Tuple[Tensor, ...]], out: Tensor, grad: Tensor
) -> List[Tensor]:
    assert out.backfn is not None
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
            oldgrad.mutate(oldgrad.data + accumgrad.data)
        if nodes:
            items = [[n, g] for n, g in zip(nodes, node.apply(grad))]
            queue.extend(items)
    return list(t.const() for t in inptmap.values())


def vjp(
    inpt: Union[Tensor, Tuple[Tensor, ...]],
    out: Tensor,
    cots: Union[Tesnor, Tuple[Tensor, ...]],
    fn,
):
    pass


def mismatch(tensor, grad):
    return tensor.dim != grad.dim and tensor.ndim <= grad.dim


def sumgrad(tensor, grad):
    dims = sumdims(tensor.dim, grad.dim, tensor.ndim, grad.ndim)
    keepdims = tensor.ndim == grad.ndim
    data = np.sum(grad.data, axis=dims, keepdims=keepdims)
    return grad.mutated(data=data)


def sumdims(tdim, gdim, tndim, gndim):
    paddim = np.pad(tdim, (gndim - tndim, 0), constant_values=0)
    mask = paddim != np.array(gdim)
    return np.where(mask)


def mapify(inpt):
    if deepnet.istensor(inpt):
        inpt = (inpt,)
    return {t: deepnet.zeroslike(t).mut() for t in inpt}
