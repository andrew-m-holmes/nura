import nura
from typing import Tuple, Optional


class Node:

    def __init__(self, function, context, accumulate, outputs):
        self._function = function
        self._context = context
        self._accumulate = accumulate
        self._outputs = outputs

    @property
    def function(self):
        return self._function

    @property
    def context(self):
        return self._context

    @property
    def accumulate(self) -> bool:
        return self._accumulate

    @property
    def outputs(self) -> int:
        return self._outputs

    def nextfunctions(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        raise NotImplementedError

    def __repr__(self) -> str:
        fn = self.function.name() if self.function is not None else None
        return f"{self.__class__.__name__}({fn=})"


def totensor(rawout):
    tensor = (
        tuple(nura.tensor(ro) for ro in rawout)
        if isinstance(rawout, tuple)
        else nura.tensor(rawout)
    )
    return tensor


def getnextfunctions(out, function, context) -> Tuple[Tuple[Optional[Node], int], ...]:
    raise NotImplementedError


def genout(rawout, function, context):
    out = totensor(rawout)
    if not context.usesgrad():
        return out
    if nura.reversemode():
        return rmout(out, function, context)
    if nura.forwardmode():
        return fmout(out, function, context)
