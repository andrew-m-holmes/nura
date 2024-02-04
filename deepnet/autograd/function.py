import deepnet
from . import graph
from deepnet.tensors import Tensor
from typing import Tuple, Union, Any, Optional
from numpy import ndarray


class Context:

    def __init__(self, fn) -> None:
        self._fn: Function = fn
        self._tensors: Optional[Tuple[Tensor, ...]] = None

    def save(self, *tensors: Tensor):
        self._tensors = tensors

    def tensors(self) -> Union[Tensor, Tuple[Tensor, ...], None]:
        if self._tensors is None:
            return None
        return self._tensors if len(self._tensors) > 1 else self._tensors[0]

    def apply(self, *args: Tensor, rev=True):
        if rev:
            return self._fn.backward(self, *args)
        return self._fn.jvp(self, *args)

    @property
    def fname(self):
        return self._fn.__name__

    def __repr__(self) -> str:
        return f"{self.fname}ctx"


class Function:

    @staticmethod
    def forward(ctx: Context, *args: Union[Tensor, Any], **kwargs: Any) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def jvp(ctx: Context, *grads: Tensor) -> ndarray:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Union[Tensor, Any], **kwargs: Any) -> Tensor:
        ctx = Context(cls)
        rawout = cls.forward(ctx, *args, **kwargs)
        irout = deepnet.tensor(rawout)
        out = graph.genout(irout, ctx)
        return out
