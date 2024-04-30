from typing import Tuple, Optional


class Node:

    def __init__(self, function, context, nextfunctions, outputs):
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
    def outputs(self) -> int:
        return self._outputs

    @property
    def nextfunctions(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        raise NotImplementedError

    def __repr__(self) -> str:
        fn = self.function.name() if self.function is not None else None
        return f"{self.__class__.__name__}({fn=})"


class AccumulateGrad:

    @staticmethod
    def apply(tensor, grad):
        if tensor.dim != grad.dim and tensor.ndim <= grad.ndim:
            pass


def getnextfunctions(function, context) -> Tuple[Tuple[Optional[Node], int], ...]:
    if not context.usesgrad():
        raise ValueError("Received context that does not use gradients")

    nextfunctions = []
    for t in context.tensors():
        pass


def genout(out, function, context):
    return out
