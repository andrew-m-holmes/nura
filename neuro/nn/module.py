import neuro.types as types
from neuro.types import dtype
from neuro.nn.parameter import Parameter, param
from neuro.tensors import Tensor
from collections import OrderedDict
from typing import Optional, Type, Iterator, Tuple, Any


class Module:

    def __init__(self) -> None:
        self._mods: OrderedDict[str, "Module"] = OrderedDict()
        self._params: OrderedDict[str, Parameter] = OrderedDict()
        self._training: bool = True
        self._dtype: Optional[Type[dtype]] = None

    @property
    def training(self) -> bool:
        return self._training

    @property
    def dtype(self):
        return self._dtype

    def forward(self) -> Any:
        raise NotImplemented

    def mods(self) -> Iterator["Module"]:
        yield from self._mods.values()
        for m in self._mods.values():
            yield from m.mods()

    def namedmods(self) -> Iterator[Tuple[str, "Module"]]:
        yield from self._mods.items()
        for m in self._mods.values():
            yield from m.namedmods()

    def params(self) -> Iterator[Parameter]:
        yield from self._params.values()
        for m in self._mods.values():
            yield from m.params()

    def namedparams(self) -> Iterator[Tuple[str, Parameter]]:
        yield from self._params.items()
        for m in self._mods.values():
            yield from m.namedparams()

    def param(self, a: Tensor, dtype: Optional[Type[types.dtype]] = None) -> Parameter:
        return param(a, True, dtype)

    def train(self):
        self._training = True
        return self

    def eval(self):
        self._training = False
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._mods[name] = value
        if isinstance(value, Parameter):
            self._params[name] = value
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.xrepr()

    def xrepr(self) -> str:
        strs = [self.__class__.__name__, "\n"]
        for n, m in self._mods.items():
            strs.append(f"{n}: ")
            strs.extend(m.xrepr())
        return "".join(strs)
