import numpy as np
import deepnet as dn
from deepnet.tensors import Tensor
from typing import Dict, Generator, Tuple, Union, Optional, Callable, Any
from collections import deque


def backward(out: Tensor, grad: Optional[Tensor] = None) -> None:
    assert out.backfn is not None
    if grad is None:
        assert out.nelem == 1
        grad = dn.oneslike(out)
    queue = deque()
    queue.append([out.backfn, grad])

    while queue:
        node, grad = queue.popleft()
        nodes = node.children()
        tensor = node.tensor
        if tensor.leaf:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = tensor.grad if tensor.grad is not None else dn.zeroslike(tensor)
            newgrad = oldgrad.mutated(data=(oldgrad.data + accumgrad.data))
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
        grad = dn.oneslike(out)
    inpt = tupify(inpt)
    vals = tuple(dn.zeroslike(t) for t in inpt)
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


def mapify(inpt, vals) -> Dict[Tensor, Any]:
    return {t: v for t, v in zip(inpt, vals)}


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


def tupify(inpt) -> Tuple[Tensor, ...]:
    if dn.istensor(inpt):
        return (inpt,)
    return inpt


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
    vec = vec.mutated(usegrad=False, grad=None)
    with dn.autograd(enabled=True, rev=True):
        out = f(*inpt, *args, **kwargs)
    out.backward(vec)
    grads = tuple(t.grad for t in inpt)
    return out.mutated(usegrad=False, leaf=True), grads


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
    assert all(v.gradtensor() for v in vec)
    inpt = tuple(t.mutated(usegrad=True, grad=g) for t, g in zip(inpt, vec))
    with dn.autograd(enabled=True, rev=False):
        out = f(*inpt, *args, **kwargs)
    grad = out.grad
    return out.mutated(usegrad=False, leaf=True), grad


def jacrev(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos: int,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inpt = tupify(inpt)
    with dn.autograd(enabled=False):
        out = f(*inpt, *args, **kwargs)
    tensor = inpt[pos]
    jac = getjac(tensor, out)
    perts = getperts(out)
    for row, pert in zip(np.ndindex(out.dim), perts):
        _, grads = vjp(inpt, pert, f, *args, **kwargs)
        jacrow = grads[pos]
        jac[row, ...] = jacrow
    return out, jac


def jacfwd(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos: int,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inpt = tupify(inpt)
    with dn.autograd(enabled=False):
        out = f(*inpt, *args, **kwargs)
    tensor = inpt[pos]
    perts = getperts(tensor)
    jac = getjac(tensor, out)
    left = tuple(dn.zeroslike(inpt[i]) for i in range(pos))
    right = tuple(dn.zeroslike(inpt[i]) for i in range(pos + 1, len(inpt)))
    for col, pert in zip(np.ndindex(tensor.dim), perts):
        vec = left + (pert,) + right
        _, jaccol = jvp(inpt, vec, f, *args, **kwargs)
        jac[..., *col] = jaccol
    return out, jac


def getperts(tensor: Tensor) -> Generator[Tensor, None, None]:
    nelem, dim, dtype = tensor.nelem, tensor.dim, tensor.dtype
    perts = dn.zeros((nelem,) + dim).to(dtype)
    arange = np.arange(nelem)
    indices = np.unravel_index(arange, dim)
    perts[arange, *indices] = 1.0
    return (perts[i] for i in range(nelem))


def getjac(tensor: Tensor, out: Tensor) -> Tensor:
    dim = out.dim + tensor.dim
    jac = dn.zeros(dim).to(out.dtype)
    return jac
