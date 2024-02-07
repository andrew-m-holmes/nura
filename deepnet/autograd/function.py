import deepnet
from deepnet.tensors import Tensor
from deepnet.autograd.graph import genout
from typing import Tuple, Union, Any, Optional
from numpy import ndarray


class Context:

    def __init__(self) -> None:
        self._tensors: Optional[Tuple[Tensor, ...]] = None

    def save(self, *tensors: Tensor):
        self._tensors = tensors

    def tensors(self) -> Optional[Union[Tensor, Tuple[Tensor, ...]]]:
        if self._tensors is None:
            return None
        return self._tensors if len(self._tensors) > 1 else self._tensors[0]

    def __repr__(self) -> str:
        return self.__class__.__name__


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
        ctx = Context()
        rawout = cls.forward(ctx, *args, **kwargs)
        irout = deepnet.tensor(rawout)
        out = genout(irout, cls, ctx)
        return out
