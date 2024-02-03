import deepnet
import numpy as np
from .mode import usegrad


def genout(out, ctx):
    node = Node(out, ctx) if usegrad() and candiff(ctx) else None
    return out.mutated(backfn=node, mut=True, leaf=False)


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
        rawgrad = self.ctx.apply(grad)
        if isinstance(rawgrad, np.ndarray):
            rawgrad = (rawgrad,)
        return tuple(deepnet.tensor(arr) for arr in rawgrad)

    def nxtnodes(self):
        if self.ctx:
            return [getnode(t) for t in self.ctx.tensors()]
        return None

    def __call__(self, *args):
        return self.apply(*args)

    def __repr__(self):
        return self._ctx.__class__.__name__


def getnode(tensor):
    if tensor.leaf:
        return Node(tensor, None)
    return tensor.backfn


def candiff(ctx):
    if ctx:
        return all(t.dtype.candiff() for t in ctx.tensors()) and any(
            t.mut for t in ctx.tensors()
        )
    return None
