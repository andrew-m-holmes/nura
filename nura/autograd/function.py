import nura
from nura.tensors import Tensor
from numpy import ndarray
from typing import Tuple, Any, Optional, Union, OrderedDict


class Context:

    def __init__(self) -> None:
        self._context: Optional[Tuple[Tuple[Tensor, int], ...]] = None

    def save(self, *tensors: Tensor) -> None:
        self._context = tuple((t, t.version) for t in tensors)

    def tensors(self) -> Tuple[Tensor, ...]:
        if self._context is None:
            return ()
        if not all(t.version == v for t, v in self._context):
            raise RuntimeError(
                "Cannot retrieve tensors, one or more tensor's version(s) has changed between initial save and retrieval"
            )
        return tuple(t for t, _ in self._context)

    def usesgrad(self) -> bool:
        if self._context is None:
            return False
        return any(t.usegrad for t, _ in self._context) and all(
            t.gradtensor for t, _ in self._context
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
    def backward(
        context: Context, *args: Any, **kwargs: Any
    ) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @staticmethod
    def tangent(context: Context, *args: Any, **kwargs: Any) -> ndarray:
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
