import nura
from numpy import ndarray
from typing import List
from collections import deque


class Node:

    def __init__(self, tensor, function, context, retain):
        self._tensor = tensor
        self._function = function
        self._context = context
        self._retain = retain

    @property
    def tensor(self):
        return self._tensor

    @property
    def function(self):
        return self._function

    @property
    def context(self):
        return self._context

    @property
    def retain(self) -> bool:
        return self._retain

    def apply(self, *grad):
        out = self.function.backward(self.context, *grad)

    def children(self) -> List["Node"]:
        if self.context is None:
            return []
        nodes = [getnode(t) for t in self.context.tensors()]
        return nodes

    def accumulate(self):
        return self.tensor.usegrad and self.retain

    def __repr__(self) -> str:
        if self.function is not None:
            return f"{self.__class__.__name__}({self.function.__name__})"
        if self.accumulate():
            return f"{self.__class__.__name__}(Accumulate)"
        return f"{self.__class__.__name__}()"


def getnode(tensor) -> Node:
    if tensor.backfn is not None:
        return tensor.backfn
    return Node(tensor, None, None, tensor.leaf)


def genout(out, function, context):
    # TODO figure out what you want to do with modes
    if not context.usesgrad():
        return out
    if nura.usegrad() and nura.reversemode():
        return rmout(out, function, context)
    if nura.usegrad() and nura.forwardmode():
        return fmout(out, function, context)


def rmout(out, function, context):
    if not isinstance(out, tuple):
        node = Node(out, function, context, False)
        out.mutate(backfn=node, usegrad=True, leaf=False, graph=1)
        return out
    nodes = [Node(o, function, context, False) for o in out]
    return tuple(
        o.mutate(backfn=n, usegrad=True, leaf=False, graph=1)
        for o, n in zip(out, nodes)
    )


def getgrads(context):
    return tuple(
        t.grad if t.grad is not None else nura.zeroslike(t) for t in context.tensors()
    )


def fmout(out, function, context):
    inputgrads = getgrads(context)
    if not isinstance(out, tuple):
        grad = nura.tensor(function.tangent(*inputgrads))
        out.mutate(usegrad=True, grad=grad, leaf=False)
        return out
    grads = [nura.tensor(g) for g in function.tangent(*inputgrads)]
    return tuple(o.mutate(usegrad=True, grad=g, leaf=False) for o, g in zip(out, grads))
