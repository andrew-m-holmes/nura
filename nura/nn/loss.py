import nura.nn.functional as f
from nura.tensors import Tensor
from typing import Optional, Any


class Loss:

    def __init__(self, reduction: Optional[str]) -> None:
        self._reduction = reduction

    @property
    def reduction(self) -> Optional[str]:
        return self._reduction

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        reduction = self.reduction
        return f"{self.name()}({reduction=})"


class CrossEntropy(Loss):

    def __init__(
        self, ignoreid: Optional[int] = None, reduction: Optional[str] = "mean"
    ):
        super().__init__(reduction)
        self._ignoreid = ignoreid

    @property
    def ignoreid(self) -> Optional[int]:
        return self._ignoreid

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return f.crossentropy(x, y, ignoreid=self.ignoreid)

    def __repr__(self) -> str:
        ignoreid, reduction = self.ignoreid, self.reduction
        return f"{self.name()}({ignoreid=} {reduction=})"


class BinaryCrossEntropy(Loss):

    def __init__(self, reduction: Optional[str] = "mean"):
        super().__init__(reduction)

    def forward(self, a: Tensor, y: Tensor) -> Tensor:
        return f.binarycrossentropy(a, y)


class MSE(Loss):

    def __init__(self, reduction: Optional[str] = "mean"):
        super().__init__(reduction)

    def forward(self, a: Tensor, y: Tensor) -> Tensor:
        return f.mse(a, y)
