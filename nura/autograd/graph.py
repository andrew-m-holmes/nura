from typing import Optional, Tuple, List
from collections import defaultdict


class Node:

    def __init__(self, function, context=None, edges=None, outputs: int = 0):
        self._function = function
        self._context = context
        self._edges = edges
        self._outputs = outputs

    @property
    def function(self):
        return self._function

    @property
    def context(self):
        return self._context

    @property
    def outputs(self) -> int:
        return self._outputs

    @property
    def edges(self) -> Optional[Tuple[Tuple["Node", int], ...]]:
        return self._edges

    def __repr__(self) -> str:
        fn = self.function.name()
        return f"{self.__class__.__name__}({fn=})"


class Accumulate:

    def __init__(self, tensor) -> None:
        self._tensor = tensor

    def backward(self, grad):
        pass

    @classmethod
    def name(cls) -> str:
        return cls.__name__


def topological(node: Node) -> List[Optional[Node]]:
    graph = {}


def getedges(context) -> Tuple[Tuple[Optional[Node], int], ...]:
    return tuple((getnode(t), t.index) for t in context.tensors())


def getnode(tensor) -> Optional["Node"]:
    if tensor.leaf and tensor.usegrad and tensor.gradfn is None:
        node = Node(Accumulate(tensor), outputs=1)
        tensor.mutate(gradfn=node)
    return tensor.gradfn


def addtograph(outputs, function, context) -> None:
    edges = getedges(context)
    if isinstance(outputs, tuple):
        node = Node(function, context, edges, len(outputs))
        for o in outputs:
            o.mutate(gradfn=node, usegrad=True, leaf=False)
    else:
        node = Node(function, context, edges, 1)
        outputs.mutate(gradfn=node, usegrad=True, leaf=False)
