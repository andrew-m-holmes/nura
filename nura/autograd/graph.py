from numpy import ndarray
from typing import Optional, Tuple, List, Dict, Union
from collections import deque


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

    def apply(self, *grad) -> Union[Tuple[ndarray, ...], ndarray]:
        return self.function.backward(self.context, *grad)

    def __repr__(self) -> str:
        fn = self.function.name()
        return f"{self.__class__.__name__}({fn=})"


class Accumulate:

    def __init__(self, tensor) -> None:
        self._tensor = tensor

    def backward(self, context, grad) -> None:
        if self._tensor.grad is None:
            self._tensor.zerograd()
        self._tensor._grad += grad
        return None

    @classmethod
    def name(cls) -> str:
        return cls.__name__


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


def constructgraph(nodes: Tuple[Node, ...]) -> Dict[Node, List[Node,]]:
    graph = dict()
    visit = set(nodes)
    queue = deque(visit)

    while queue:
        node = queue.popleft()
        if node not in graph:
            graph[node] = []
        if node.edges is None:
            continue
        for n, i in node.edges:
            if n is not None:
                graph[node].append(n)
                if n not in visit:
                    queue.append(n)
                    visit.add(n)
    return graph


def topological(graph: Dict[Node, List[Node]]) -> List[Node]:
    indegree = {n: 0 for n in graph.keys()}
    for node, edges in graph.items():
        for child in set(edges):
            indegree[child] += 1 * node.outputs

    order = []
    queue = deque([n for n in indegree.keys() if not indegree[n]])
    while queue:
        node = queue.popleft()
        order.append(node)
        edges = graph[node]
        for child in set(edges):
            indegree[child] -= 1 * node.outputs
            if not indegree[child]:
                queue.append(child)
    return order
