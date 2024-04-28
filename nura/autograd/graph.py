import nura
from typing import Tuple, Optional


class Node:

    def __init__(self, function, context, accumulate, outputs):
        self._function = function
        self._context = context
        self._accumulate = accumulate
        self._outputs = outputs

    @property
    def function(self):
        return self._function

    @property
    def context(self):
        return self._context

    @property
    def accumulate(self) -> bool:
        return self._accumulate

    @property
    def outputs(self):
        return self._outputs

    def nextfunctions(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        return tuple((getnode(t), t.index) for t in self.context.tensors())

    def __repr__(self) -> str:
        fn = self.function.name() if self.function is not None else None
        accumulate = self.accumulate
        return f"{self.__class__.__name__}({fn=} {accumulate=})"


def getnode(tensor) -> Optional[Node]:
    if tensor.usegrad and tensor.leaf:
        return Node(None, None, True, 0)
    return tensor.gradfn


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
        node = Node(function, context, False, 1)
        out.mutate(gradfn=node, usegrad=True, leaf=False)
        return out
    node = Node(function, context, False, len(out))
    for i, o in enumerate(out):
        o.mutate(gradfn=node, usegrad=True, leaf=False, index=i)
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
