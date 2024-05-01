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

    def __repr__(self) -> str:
        fn = self.function.name() if self.function is not None else None
        return f"{self.__class__.__name__}({fn=})"
