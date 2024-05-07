import nura
from nura.tensors import Tensor
from numpy import ndarray
from typing import Tuple, Any, Optional, Union, OrderedDict


class Context:

    def __init__(self) -> None:
        self._context: Optional[OrderedDict[Tensor, int]] = None

    def save(self, *tensors: Tensor) -> None:
        self._context = OrderedDict((t, t.version) for t in tensors)

    def tensors(self) -> Tuple[Tensor, ...]:
        return tuple(self._context.keys()) if self._context is not None else ()

    def usesgrad(self) -> bool:
        if self._context is None:
            return False
        return any(t.usegrad for t in self._context.keys()) and all(
            t.gradtensor for t in self._context.keys()
        )

    def __getattr__(self, name: str) -> Any:
        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.__class__.__name__


class Function:

    @staticmethod
    def forward(context: Context, *args: Any, **kwargs: Any) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(context: Context, grad: Tensor) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @staticmethod
    def tangent(
        context: Context, *grads: Tensor
    ) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        context = Context()
        arr = cls.forward(context, *args, **kwargs)
        output = nura.tensor(arr)
        if context.usesgrad() and nura.Autograd.enabled():
            nura.graph.addtograph(output, cls, context)
        return output

    @classmethod
    def name(cls) -> str:
        return cls.__name__
