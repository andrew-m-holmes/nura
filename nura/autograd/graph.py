from typing import Optional
from nura.interface import Tensor, Function, Context


class Node:

    def __init__(
        self,
        function: Optional[Function] = None,
        context: Optional[Context] = None,
        outputs: int = -1,
    ):
        self._function = function
        self._context = context
        self._outputs = outputs

    @property
    def function(self) -> Optional[Function]:
        return self._function

    @property
    def context(self) -> Optional[Context]:
        return self._context

    @property
    def outputs(self) -> int:
        return self._outputs

    def __repr__(self) -> str:
        fn = self.function.name() if self.function is not None else None
        return f"{self.__class__.__name__}({fn=})"


def getnode(tensor: Tensor):
    if tensor.leaf and tensor.usegrad:
        return Node()
    return tensor.gradfn


def addtograph(outputs: Tensor, function: Function, context: Context):
    pass
