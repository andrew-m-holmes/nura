import nura
from typing import List, Iterator, Tuple, Optional


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

    def apply(self, *grad) -> Iterator[Tuple]:
        for t, o in zip(self.context.tensors(), self.function.backward(*grad)):
            node, grad = getnode(t), nura.tensor(o)
            yield node, grad

    def children(self) -> List["Node"]:
        if self.context is None:
            return []
        nodes = [getnode(t) for t in self.context.tensors()]
        return nodes

    def topological(self, mem: Optional[List["Node"]] = None) -> List["Node"]:
        if mem is None:
            mem = []
        children = self.children()
        mem.extend(self.children())
        for c in children:
            c.topological(mem)
        return mem

    def accumulate(self):
        return self.tensor.usegrad and self.retain

    def __repr__(self) -> str:
        if self.function is not None:
            return f"{self.__class__.__name__}({self.function.name()})"
        if self.accumulate():
            return f"{self.__class__.__name__}(Accumulate)"
        return f"{self.__class__.__name__}()"


def getnode(tensor) -> Node:
    if tensor.backfn is not None:
        return tensor.backfn
    return Node(tensor, None, None, tensor.leaf)


def totensor(rawout):
    tensor = (
        tuple(nura.tensor(ro) for ro in rawout)
        if isinstance(rawout, tuple)
        else nura.tensor(rawout)
    )
    return tensor


def genout(rawout, function, context):
    out = totensor(rawout)
    if not context.usesgrad():
        return out
    if nura.reversemode():
        return rmout(out, function, context)
    if nura.forwardmode():
        return fmout(out, function, context)


def rmout(out, function, context):
    if not isinstance(out, tuple):
        node = Node(out, function, context, False)
        out.mutate(backfn=node, usegrad=True, leaf=False, graph=1)
        return out
    nodes = [Node(o, function, context, False) for o in out]
    for o, n in zip(out, nodes):
        o.mutate(backfn=n, usegrad=True, leaf=False, graph=1)
    return out


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
