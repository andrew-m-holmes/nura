import nura
from typing import List, Optional


class Node:

    def __init__(self, tensor, function, context, retain):
        self._tensor = tensor if isinstance(tensor, tuple) else (tensor,)
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

    @property
    def outputs(self) -> int:
        if isinstance(self.tensor, tuple):
            return len(self.tensor)
        return 1

    def indexof(self, tensor) -> int:
        for i in range(self.outputs):
            if self.tensor[i] is tensor:
                return i
        return -1

    def apply(self, *grad):
        out = self.function.backward(self.context, *grad)
        return (
            tuple(nura.tensor(o) for o in out)
            if isinstance(out, tuple)
            else nura.tensor(out)
        )

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
        fn = self.function.name() if self.function is not None else None
        accumulate = self.accumulate()
        return f"{self.__class__.__name__}({fn=} {accumulate=})"


def getnode(tensor) -> Node:
    if tensor.backfn is not None:
        return tensor.backfn
    return Node(tensor, None, None, tensor.leaf, -1)


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
    node = Node(out, function, context, False, 0)
    if not isinstance(out, tuple):
        out.mutate(backfn=node, usegrad=True, leaf=False, graph=1)
        return out
    for o in out:
        o.mutate(backfn=node, usegrad=True, leaf=False, graph=1)
    return out


def fmout(out, function, context):
    inputgrads = tuple(
        t.grad if t.grad is not None else nura.zeroslike(t) for t in context.tensors()
    )
    if not isinstance(out, tuple):
        grad = nura.tensor(function.tangent(*inputgrads))
        out.mutate(usegrad=True, grad=grad, leaf=False)
        return out
    grads = (nura.tensor(g) for g in function.tangent(*inputgrads))
    return tuple(o.mutate(usegrad=True, grad=g, leaf=False) for o, g in zip(out, grads))
