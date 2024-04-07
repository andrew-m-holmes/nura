import nura.nn.functional as f
from nura.nn.module import Module
from typing import Optional
from nura.tensors import Tensor


class CrossEntropy(Module):

    def __init__(self, ignoreid: Optional[int] = None):
        super().__init__()
        self._ignoreid = ignoreid

    @property
    def ignoreid(self) -> Optional[int]:
        return self._ignoreid

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return f.crossentropy(x, y, ignoreid=self.ignoreid)

    def xrepr(self) -> str:
        ignoreid = self.ignoreid
        return f"{self.name()}({ignoreid=})"


class BinaryCrossEntropy(Module):

    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, y: Tensor) -> Tensor:
        return f.binarycrossentropy(a, y)
