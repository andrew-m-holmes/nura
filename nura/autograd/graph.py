from typing import Optional, Tuple


class Node:

    def __init__(
        self, function, context=None, nextfunctions=None, outputs: Optional[int] = None
    ):
        self._function = function
        self._context = context
        self._nextfunctions = nextfunctions
        self._outputs = outputs

    @property
    def function(self):
        return self._function

    @property
    def context(self):
        return self._context

    @property
    def outputs(self) -> Optional[int]:
        return self._outputs

    @property
    def nextfunctions(self) -> Optional[Tuple[Tuple["Node", int], ...]]:
        return self._nextfunctions

    def __repr__(self) -> str:
        fn = self.function.name()
        return f"{self.__class__.__name__}({fn=})"


class Accumulate:

    def __init__(self, tensor) -> None:
        self._tensor = tensor

    def backward(self, grad):
        pass


def getnextfunctions(context) -> Tuple[Tuple[Optional[Node], int], ...]:
    return tuple((getnode(t), t.index) for t in context.tensors())


def getnode(tensor) -> Optional["Node"]:
    if tensor.leaf and tensor.usegrad and tensor.gradfn is None:
        node = Node(Accumulate(tensor))
        tensor.mutate(gradfn=node)
    return tensor.gradfn


def addtograph(outputs, function, context) -> None:
    if isinstance(outputs, tuple):
        node = Node(function, context, len(outputs))
        for o in outputs:
            o.mutate(gradfn=node, usegrad=True, leaf=False)
    else:
        node = Node(function, context, 1)
        outputs.mutate(gradfn=node, usegrad=True, leaf=False)
