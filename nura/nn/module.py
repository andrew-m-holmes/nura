import nura.types as types
from nura.types import dtype
from nura.nn.parameter import Parameter
from collections import OrderedDict
from typing import Type, Iterator, Tuple, Any
from copy import copy


class Module:

    def __init__(self) -> None:
        self._mods: OrderedDict[str, "Module"] = OrderedDict()
        self._params: OrderedDict[str, Parameter] = OrderedDict()
        self._training: bool = True

    @property
    def training(self) -> bool:
        return self._training

    @classmethod
    def name(cls) -> str:
        return cls.__name__

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

    def to(self, dtype: Type[dtype]) -> "Module":
        params = OrderedDict((n, p.to(dtype)) for n, p in self._params.items())
        mods = OrderedDict((n, m.to(dtype)) for n, m in self._mods.items())
        mod = copy(self)
        mod._params = params
        mod._mods = mods
        return mod

    def half(self):
        return self.to(types.half)

    def float(self):
        return self.to(types.float)

    def double(self):
        return self.to(types.double)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._mods[name] = value
        if isinstance(value, Parameter):
            self._params[name] = value
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.repr()

    def repr(self, pad=3) -> str:
        strs = [self.xrepr()]
        if hasmods := len(self._mods):
            strs.append(": (")
        strs.append("\n")
        for n, m in self._mods.items():
            strs.append(f"{' ' * pad}({n}): ")
            strs.extend(m.repr(pad + 3))
        if hasmods:
            strs.append(f"{' ' * (pad - 3)})\n")
        return "".join(strs)

    def xrepr(self) -> str:
        return f"{self.__class__.__name__}()"
