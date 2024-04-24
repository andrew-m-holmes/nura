import nura
from typing import List, Optional


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

    def accumulate(self):
        return self.tensor.usegrad and self.retain

    def __repr__(self) -> str:
        if self.function is not None:
            return f"{self.__class__.__name__}({self.function.__name__})"
        if self.accumulate():
            return f"{self.__class__.__name__}(Accumulate)"
        return f"{self.__class__.__name__}()"


def getnode(tensor) -> Node:
    if tensor.leaf and tensor.usegrad:
        return Node(tensor, None, None, True)
    if tensor.backfn is not None:
        return tensor.backfn
    return Node(tensor, None, None, False)


def getgrads(context):
    return tuple(
        t.grad if t.grad is not None else nura.zeroslike(t) for t in context.tensors()
    )


def genout(out, function, context):
    if not context.usesgrad():
        return out
    node = Node(out, function, context, False)
    if nura.usegrad() and nura.reversemode():
        out.mutate(backfn=node, usegrad=True, leaf=False, graph=1)
    elif nura.usegrad() and nura.forwardmode():
        grads = getgrads(context)
        grad = node.apply(*grads, backward=False)
        out.mutate(usegrad=True, grad=grad, leaf=False)
    return out
