import deepnet
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
            return tuple(deepnet.tensor(a) for a in arr)
        arr = self.function.tangent(self.context, *grad)
        return deepnet.tensor(arr)

    def children(self) -> Optional[List["Node"]]:
        if self.context is None:
            return None
        nodes = []
        for t in self.context.tensors():
            node = getnode(t)
            if isinstance(node, Node):
                nodes.append(node)
        return nodes

    def __repr__(self):
        if self.tensor.leaf:
            return "accumgrad"
        return f"{self.function.__name__.lower()}"


def getnode(tensor):
    if tensor.leaf and tensor.usegrad:
        return Node(tensor, None, None)
    return tensor.backfn


def genout(out, function, context):
    if not context.usesgrad():
        return out
    node = Node(out, function, context)
    if deepnet.usegrad() and deepnet.reversemode():
        out.mutate(backfn=node, usegrad=True, leaf=False)
    elif deepnet.usegrad() and deepnet.forwardmode():
        grads = getgrads(context)
        grad = node.apply(*grads, backward=False)
        out.mutate(usegrad=True, grad=grad, leaf=False)
    return out


def getgrads(context):
    return tuple(
        t.grad if t.grad is not None else deepnet.zeroslike(t)
        for t in context.tensors()
    )


