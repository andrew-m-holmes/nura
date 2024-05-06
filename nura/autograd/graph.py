import numpy as np
import nura
from nura.tensors import Tensor
from nura.autograd.function import Function, Context
from typing import Optional, Type, Tuple, List, Union, Callable, DefaultDict, Dict
from collections import deque, defaultdict


class Node:

    def __init__(
        self,
        outputs: Tuple[Tensor, ...],
        edges: Optional[Tuple[Tuple[Optional["Node"], int], ...]],
        closure: Optional[Callable[..., Tuple[Tensor, ...]]],
    ):
        self._outputs = outputs
        self._closure = closure
        self._edges = edges
        self._retained_outputs = None

    @property
    def outputs(self) -> Tuple[Tensor, ...]:
        return self._outputs

    @property
    def edges(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        if self._edges is None:
            return ()
        return self._edges

    def retain(self, index: int):
        if self._retained_outputs is None:
            self._retained_outputs = [False] * len(self._outputs)
        self._retained_outputs[index] = True

    def unretain(self, index: int):
        if self._retained_outputs is None:
            raise RuntimeError(
                "Cannot remove gradient accumulation, tensor never retained gradient"
            )
        self._retained_outputs[index] = False

    def apply(self, *grads: Tensor) -> Tuple[Tensor, ...]:
        if self._closure is None:
            raise RuntimeError("Cannot apply closure function, closure does not exist")
        return self._closure(*grads)

    def accumulate(self, *grads: Tensor):
        if self._retained_outputs is None:
            raise RuntimeError("Cannot accumulate gradient, no outputs retain gradient")
        for output in self._outputs:
            grad = grads[output.index]
            if self._retained_outputs[output.index]:
                if output._grad is None:
                    output._grad = nura.zeroslike(output)
                output._grad += grad

    def name(self) -> str:
        name = self._closure.__name__ if self._closure is not None else "Accumulate"
        return name

    def __repr__(self) -> str:
        return f"Node({self.name()})"


def add_to_graph(
    outputs: Union[Tuple[Tensor, ...], Tensor],
    function: Type[Function],
    context: Context,
) -> None:
    edges = tuple((get_node(t), t.index) for t in context.tensors())
    closure = define_closure(function, context)
    if isinstance(outputs, tuple):
        node = Node(outputs, edges, closure)
        for output in outputs:
            output.mutate(usegrad=True, gradfn=node, leaf=False)
    else:
        node = Node((outputs,), edges, closure)
        outputs.mutate(usegrad=True, gradfn=node, leaf=False)


def define_closure(
    function: Type[Function], context: Context
) -> Callable[..., Tuple[Tensor, ...]]:
    def closure(*grads: Tensor) -> Tuple[Tensor, ...]:
        grad_arrays = function.backward(context, *grads)
        new_grads = function.array_to_tensors(grad_arrays)
        return new_grads if isinstance(new_grads, tuple) else (new_grads,)

    closure.__name__ = f"{function.name()}Backward"
    return closure


def get_node(tensor: Tensor) -> Optional[Node]:
    if tensor.leaf and tensor.usegrad and tensor.gradfn is None:
        node = Node((tensor,), None, None)
        node.retain(0)
        tensor.mutate(gradfn=node)
    return tensor.gradfn


def construct_graph(
    nodes: Union[Tuple[Node, ...], Node]
) -> DefaultDict[Node, List[Node]]:
    if isinstance(nodes, Node):
        nodes = (nodes,)
    graph = defaultdict(list)
    visit = set(nodes)
    queue = deque(visit)

    while queue:
        node = queue.popleft()
        graph[node]
        for child_node, _ in node.edges:
            if child_node is None:
                continue
            graph[node].append(child_node)
            if child_node not in visit:
                queue.append(child_node)
                visit.add(child_node)
    return graph


def topological(graph: Dict[Node, List[Node]]) -> Tuple[Node, ...]:
    indegree = defaultdict.fromkeys(graph.keys(), 0)
    for node, edges in graph.items():
        for child_node in set(edges):
            indegree[child_node] += 1 * len(node.outputs)

    order = []
    queue = deque([n for n in indegree.keys() if not indegree[n]])
    while queue:
        node = queue.popleft()
        order.append(node)
        edges = graph[node]
        for child_node in set(edges):
            indegree[child_node] -= 1 * len(node.outputs)
            if not indegree[child_node]:
                queue.append(child_node)
    return tuple(order)
