from collections import deque

import numpy as np

import deepnet

def backward(out, grad):
    assert out.backfn is not None
    queue = deque()
    queue.append([out.backfn, grad])

    while queue:
        node, grad = queue.popleft()
        nodes = node.nxtnodes()
        tensor = node.tensor
        if node.tensor.leaf:
            accumgrad = sumgrad(tensor, grad) if mismatch(tensor, grad) else grad
            oldgrad = tensor.grad
            newgrad = oldgrad.withstate(oldgrad.data + accumgrad.data)
            _ = tensor.withstate(newgrad)
        if nodes:
            items = [[n, g] for n, g in zip(nodes, node.apply(grad))]
            queue.extend(items)

def grad(inpt, out, grad):
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
            inptmap[tensor] = oldgrad.withstate(oldgrad.data + accumgrad.data)
        if nodes:
            items = [[n, g] for n, g in zip(nodes, node.apply(grad))]
            queue.extend(items)
    return list(inptmap.values())
        

def mismatch(tensor, grad):
    return tensor.dim != grad.dim and tensor.ndim <= grad.dim


def sumgrad(tensor, grad):
    dims = sumdims(tensor.dim, grad.dim, tensor.ndim, grad.ndim)
    keepdims = tensor.ndim == grad.ndim
    data = np.sum(grad.data, axis=dims, keepdims=keepdims)
    return grad.withstate(data=data)


def sumdims(tdim, gdim, tndim, gndim):
    paddim = np.pad(tdim, (gndim - tndim, 0), constant_values=0)
    mask = paddim != np.array(gdim)
    return np.where(mask)


def mapify(inpt):
    if deepnet.istensor(inpt):
        inpt = (inpt,)
    return {t: deepnet.zeros_like(t) for t in inpt}

