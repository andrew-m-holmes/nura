import deepnet
from numpy import ndarray
from deepnet.autograd.mode import usegrad, revmode
from typing import List, Optional


class Node:

    def __init__(self, tensor, f, ctx):
        self._tensor = tensor
        self._f = f
        self._ctx = ctx

    @property
    def tensor(self):
        return self._tensor

    @property
    def f(self):
        return self._f

    @property
    def ctx(self):
        return self._ctx

    def applyback(self, grad):
        rawgrad = self.f.backward(self.ctx, grad)
        if isinstance(rawgrad, ndarray):
            rawgrad = (rawgrad,)
        return tuple(deepnet.tensor(arr) for arr in rawgrad)

    def applytan(self, *grad):
        rawtan = self.f.tangent(self.ctx, *grad)
        return deepnet.tensor(rawtan)

    def nxtnodes(self) -> Optional[List["Node"]]:
        if self.ctx is None:
            return None
        nodes = []
        for t in self.ctx.tensors():
            node = getnode(t)
            if isinstance(node, Node):
                nodes.append(node)
        return nodes

    def __repr__(self):
        return f"{self.f.__name__.lower()}"


def genout(out, f, ctx):
    if deepnet.usegrad():
        node = Node(out, f, ctx) if usegrad() and candiff(ctx) else None
        if revmode():
            out.mutate(backfn=node, usegrad=True, leaf=False)
        elif node is not None:
            tans = gettans(ctx)
            tan = node.applytan(*tans)
            out.mutate(usegrad=True, grad=tan, leaf=False)
    return out


def getnode(tensor):
    if tensor.leaf and tensor.usegrad:
        return Node(tensor, None, None)
    return tensor.backfn


def gettans(ctx):
    return tuple(
        t.grad if t.grad is not None else deepnet.zeroslike(t) for t in ctx.tensors()
    )


def candiff(ctx):
    if ctx.tensors():
        return all(t.gradtensor() for t in ctx.tensors()) and any(
            t.usegrad for t in ctx.tensors()
        )
    return False
