import nura
from nura.tensors import Tensor
from typing import Optional, Type, Tuple, Union, Sequence
from nura.autograd.function import Function, Context
from collections import deque


class Node:

    def __init__(
        self,
        output: Tensor,
        function: Optional[Type[Function]] = None,
        context: Optional[Context] = None,
        edges: Optional[Tuple[Optional["Node"], ...]] = None,
        accumulate: bool = False,
    ) -> None:
        self._output = output
        self._function = function
        self._context = context
        self._edges = edges
        self._accumulate = accumulate

    @property
    def output(self) -> Tensor:
        return self._output

    @property
    def function(self) -> Optional[Type[Function]]:
        return self._function

    @property
    def context(self) -> Optional[Context]:
        return self._context

    @property
    def edges(self) -> Tuple[Optional["Node"], ...]:
        if self._edges is None:
            return ()
        return self._edges

    @property
    def accumulate(self) -> bool:
        return self._accumulate and self.output.usegrad

    def retain(self) -> None:
        self._accumulate = True

    def unretain(self) -> None:
        self._accumulate = False

    def apply(self, grad: Tensor) -> Union[Tuple[Tensor, ...], Tensor]:
        if self.function is None or self.context is None:
            raise RuntimeError("Cannot apply backward, function and/or context is None")
        arr = self.function.backward(self.context, grad)
        return nura.totensor(arr)

    def name(self) -> str:
        name = (
            f"{self.function.name()}Backward"
            if self.function is not None
            else "Accumulate"
        )
        return name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name()})"


def addtograph(output: Tensor, function: Type[Function], context: Context) -> None:
    edges = linkedges(context)
    node = Node(output, function, context, edges, accumulate=False)
    output.mutate(usegrad=True, gradfn=node, leaf=False)


def linkedges(context: Context) -> Tuple[Optional[Node], ...]:
    edges = []
    for t in context.tensors():
        if t.usegrad and t.leaf and t.gradfn is None:
            node = Node(t, accumulate=True)
            t.mutate(gradfn=node)
        edges.append(t.gradfn)
    return tuple(edges)


def toposort(node: Union[Sequence[Node], Node]) -> Tuple[Node, ...]:
    if not isinstance(node, Sequence):
        node = (node,)
    if not all(node):
        raise ValueError(f"Received an invalid node entry: {node}")

    visit = set(node)
    queue = deque(node)
    indegree = dict.fromkeys(node, 0)

    while queue:
        node = queue.popleft()
        for edge in node.edges:
            if edge is None:
                continue
            if edge not in visit:
                indegree[edge] = 0
                visit.add(edge)
                queue.append(edge)
            indegree[edge] += 1

    topolist = []
    queue = deque([n for n, d in indegree.items() if not d])

    while queue:
        node = queue.popleft()
        topolist.append(node)
        for edge in node.edges:
            if edge is None:
                continue
            indegree[edge] -= 1
            if not indegree[edge]:
                queue.append(edge)
    return tuple(topolist)
