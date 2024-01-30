import numpy as np


def genout(out, ctx):
    node = Node(out, ctx)
    return out.withstate(backfn=node, leaf=False) 

class Node:

    def __init__(self, tensor, ctx):
        self._tensor = tensor
        self._ctx = ctx

    @property
    def tensor(self):
        return self._tensor

    @property
    def ctx(self):
        return self._ctx

    def apply(self, grad):
        return self.ctx(grad)

    def nxtnodes(self):
        if candiff(self.ctx):
            nodes = []
            for tensor in self.ctx:
                nodes.append(getnode(tensor))
            return nodes
        return None

    def __repr__(self):
        return self._ctx.__class__.__name__

class Accumulator:

    @staticmethod
    def apply(tensor, grad):
        if tensor.dim != grad.dim and tensor.ndim <= grad.ndim:
            grad = sumgrad(tensor, grad)
        return grad

def getnode(tensor):
    if tensor.leaf:
        return Node(tensor, Accumulator)
    return tensor.backfn

def candiff(ctx):
    return any(t.diff for t in ctx.tensors())


def sumgrad(tensor, grad):
    dims = sumdims(tensor.dim, grad.dim, tensor.ndim, grad.ndim)
    keepdims = tensor.ndim == grad.ndim
    data = np.sum(grad.data, axis=dims, keepdims=keepdims)
    return grad.withstate(data=data)


def sumdims(tdim, gdim, tndim, gndim):
    paddim = np.pad(tdim, (gndim - tndim, 0), constant_values=0)
    mask = paddim != np.array(gdim)
    return np.where(mask)
