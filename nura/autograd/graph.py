from numpy import ndarray
from typing import Optional, Tuple, List, Dict, Union
from collections import deque


class Node:

    def __init__(self, function, context, edges, outputs):
        self._function = function
        self._context = context
        self._edges = edges
        self._outputs = outputs
        self._tensors = None

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
    def edges(self) -> Tuple[Tuple["Node", int], ...]:
        return self._edges

    def retain(self, tensor):
        if self._tensors is None:
            self._tensors = set()
        self._tensors.add(tensor)

    def apply(self, *grad) -> Union[Tuple[ndarray, ...], ndarray]:
        return self.function.backward(self.context, *grad)

    def __repr__(self) -> str:
        name = self.function.name() if self.function is not None else "Accumulate"
        return f"{self.__class__.__name__}({name=})"


def addtograph(output, function, context) -> None:
    edges = getedges(context)
    if isinstance(output, tuple):
        node = Node(function, context, edges, outputs=len(output))
        for o in output:
            o.mutate(gradfn=node, usegrad=True, leaf=False)
    else:
        node = Node(function, context, edges, outputs=1)
        output.mutate(gradfn=node, usegrad=True, leaf=False)


def getedges(context) -> Tuple[Tuple[Optional[Node], int], ...]:
    return tuple((getnode(t), t.index) for t in context.tensors())


def getnode(tensor) -> Optional["Node"]:
    if tensor.leaf and tensor.usegrad:
        node = Node(None, None, (), 1)
        node.retain(tensor)
        return node
    return tensor.gradfn


def constructgraph(nodes: Union[Tuple[Node, ...], Node]) -> Dict[Node, List[Node,]]:
    if isinstance(nodes, Node):
        nodes = (nodes,)
    graph = dict()
    visit = set(nodes)
    queue = deque(visit)

    while queue:
        node = queue.popleft()
        if node not in graph:
            graph[node] = []
        for n, _ in node.edges:
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
