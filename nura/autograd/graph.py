from typing import Optional, Tuple


class Node:

    def __init__(self, function=None, context=None, outputs: int = -1):
        self._function = function
        self._context = context
        self._outputs = outputs
        self._leaf = outputs == -1
        self.grads = None

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
    def leaf(self) -> bool:
        return self._leaf

    @property
    def ready(self) -> bool:
        if self.grads is None:
            return False
        return len(self.grads) == self.outputs

    @property
    def nextfunctions(self) -> Optional[Tuple[Tuple, ...]]:
        if self.context is None:
            return None
        return tuple((self.getnode(t), t.version) for t in self.context.tensors())

    @staticmethod
    def getnode(tensor) -> Optional["Node"]:
        if tensor.leaf and tensor.usegrad:
            return Node()
        return tensor.gradfn

    @staticmethod
    def accumulate(tensor, grad):
        pass

    def pushedge(self, edge, grad):
        if self.grads is None:
            self._grads = [None] * self.outputs
        self._grads[edge[1]] = grad

    def __repr__(self) -> str:
        fn = self.function.name() if self.function is not None else "_Accumulate"
        return f"{self.__class__.__name__}({fn=})"


def addtograph(outputs, function, context) -> None:
    if isinstance(outputs, tuple):
        node = Node(function, context, len(outputs))
        for o in outputs:
            o.mutate(gradfn=node, usegrad=True, leaf=False)
    else:
        node = Node(function, context, 1)
        outputs.mutate(gradfn=node, usegrad=True, leaf=False)
