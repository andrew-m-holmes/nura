from numpy import ndarray
from typing import Optional, Tuple, List, Dict, Union, Callable
from collections import deque


class Node:

    def __init__(self, outputs, edges, closure):
        self._outputs = {o.index: [o, False] for o in outputs}
        self._closure = closure
        self._edges = edges

    @property
    def outputs(self) -> Dict:
        return self._outputs

    @property
    def edges(self) -> Tuple[Tuple["Node", int], ...]:
        return self._edges

    def retain(self, tensor):
        self.outputs[tensor.index][1] = True

    def apply(self, *grad) -> Union[Tuple[ndarray, ...], ndarray]:
        return self._closure(*grad)

    def __repr__(self) -> str:
        name = self._closure.__name__ if self._closure is not None else "Accumulate"
        return f"{self.__class__.__name__}({name=})"


def addtograph(output, function, context) -> None:
    edges = tuple((get_node(t), t.index) for t in context.tensors())
    closure = get_closure(function, context)
    if isinstance(output, tuple):
        node = Node(output, edges, closure)
        for o in output:
            o.mutate(gradfn=node, usegrad=True, leaf=False)
    else:
        node = Node((output,), edges, closure)
        output.mutate(gradfn=node, usegrad=True, leaf=False)


def get_closure(function, context) -> Callable[..., Union[Tuple[ndarray], ndarray]]:
    def closure(*grad):
        return function.backward(context, *grad)

    closure.__name__ = f"{function.name()}Backward"
    return closure


def get_node(tensor) -> Optional["Node"]:
    if tensor.leaf and tensor.usegrad:
        node = Node((tensor,), (), None)
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
            indegree[child] += 1 * len(node.outputs)

    order = []
    queue = deque([n for n in indegree.keys() if not indegree[n]])
    while queue:
        node = queue.popleft()
        order.append(node)
        edges = graph[node]
        for child in set(edges):
            indegree[child] -= 1 * len(node.outputs)
            if not indegree[child]:
                queue.append(child)
    return order
