import nura
from nura.tensors import Tensor
from nura.autograd.graph import addtograph
from numpy import ndarray
from typing import Tuple, Any, Optional, Union, OrderedDict


class Context:

    def __init__(self) -> None:
        self._memory: Optional[OrderedDict[Tensor, int]] = None

    def save(self, *tensors: Tensor) -> None:
        self._memory = OrderedDict((t, t.version) for t in tensors)

    def tensors(self) -> Tuple[Tensor, ...]:
        return tuple(self._memory.keys()) if self._memory is not None else ()

    def usesgrad(self) -> bool:
        if self._memory is None:
            return False
        return any(t.usegrad for t in self._memory.keys()) and all(
            t.gradtensor for t in self._memory.keys()
        )

    def __getattr__(self, name: str) -> Any:
        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.__class__.__name__


class Function:

    @staticmethod
    def forward(
        context: Context, *args: Any, **kwargs: Any
    ) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @staticmethod
    def backward(context: Context, grad: Tensor) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @staticmethod
    def tangent(context: Context, *grad: Tensor) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        context = Context()
        rawoutput = cls.forward(context, *args, **kwargs)
        output = cls.prepoutput(rawoutput)
        if context.usesgrad():
            addtograph(output, cls, context)
        return output

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @staticmethod
    def prepoutput(
        rawoutput: Union[Tuple[ndarray, ...], ndarray]
    ) -> Union[Tuple[Tensor, ...], Tensor]:
        if isinstance(rawoutput, tuple):
            return tuple(
                nura.tensor(ro).mutated(index=i) for i, ro in enumerate(rawoutput)
            )
        return nura.tensor(rawoutput)
