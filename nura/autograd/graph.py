import nura
from nura.tensors import Tensor
from typing import Optional, Type, Tuple, Union
from nura.autograd.function import Function, Context


class Node:

    def __init__(
        self,
        output: Tensor,
        function: Optional[Type[Function]] = None,
        context: Optional[Context] = None,
        edges: Optional[Tuple[Optional["Node"], ...]] = None,
        retain: bool = False,
    ) -> None:
        self._output = output
        self._function = function
        self._context = context
        self._edges = edges
        self._retain = retain

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
    def retain(self) -> bool:
        return self._retain

    def apply(self, grad: Tensor) -> Union[Tuple[Tensor, ...], Tensor]:
        if self.function is None or self.context is None:
            raise RuntimeError("Cannot apply backward, function and/or context is None")
        arr = self.function.backward(self.context, grad)
        return nura.atot(arr)

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
    node = Node(output, function, context, edges, retain=False)
    output.mutate(usegrad=True, gradfn=node, leaf=False)


def linkedges(context: Context) -> Tuple[Optional[Node], ...]:
    edges = []
    for t in context.tensors():
        if t.usegrad and t.gradfn is None:
            node = Node(t, retain=True)
            t.mutate(gradfn=node)
        edges.append(t.gradfn)
    return tuple(edges)
