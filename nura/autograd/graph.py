import nura
from typing import List, Optional


class Node:

    def __init__(self, tensor, function, context):
        self._tensor = tensor
        self._function = function
        self._context = context

    @property
    def tensor(self):
        return self._tensor

    @property
    def function(self):
        return self._function

    @property
    def context(self):
        return self._context

    def apply(self, *grad, backward=True):
        if backward:
            arr = self.function.backward(self.context, *grad)
            if not isinstance(arr, tuple):
                arr = (arr,)
            grads = tuple(nura.tensor(a) for a in arr)
            return grads
        grad = self.function.tangent(self.context, *grad)
        return nura.tensor(grad)

    def children(self) -> Optional[List["Node"]]:
        if self.context is None:
            return None
        nodes = [getnode(t) for t in self.context.tensors()]
        return nodes

    def ongraph(self):
        return self.tensor.usegrad

    def leaf(self) -> bool:
        return self.tensor.leaf

    def __repr__(self):
        if self.function is not None:
            return f"{self.function.__name__}"
        return "Accumulate"


def getnode(tensor):
    if tensor.leaf and tensor.usegrad:
        return Node(tensor, None, None)
    return tensor.backfn


def genout(out, function, context):
    if not context.usesgrad():
        return out
    node = Node(out, function, context)
    if nura.usegrad() and nura.reversemode():
        out.mutate(backfn=node, usegrad=True, leaf=False)
    elif nura.usegrad() and nura.forwardmode():
        grads = getgrads(context)
        grad = node.apply(*grads, backward=False)
        out.mutate(usegrad=True, grad=grad, leaf=False)
    return out


def getgrads(context):
    return tuple(
        t.grad if t.grad is not None else nura.zeroslike(t) for t in context.tensors()
    )
