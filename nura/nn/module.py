import nura.types as types
from nura.types import dtype
from nura.nn.parameter import Parameter, param
from nura.tensors import Tensor
from collections import OrderedDict
from typing import Type, Iterator, Tuple, Any
from copy import copy, deepcopy


class Module:

    def __init__(self) -> None:
        self._mods: OrderedDict[str, "Module"] = OrderedDict()
        self._params: OrderedDict[str, Parameter] = OrderedDict()
        self._training: bool = True
        self._dtype: Type[dtype] = types.float

    @property
    def training(self) -> bool:
        return self._training

    @property
    def dtype(self) -> Type[dtype]:
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

    def param(self, a: Tensor) -> Parameter:
        return param(a, True, self.dtype)

    def to(self, dtype: Type[types.dtype]):
        mod = self.copy()
        mod._dtype = dtype
        mod._params = OrderedDict()
        mod._mods = OrderedDict()
        for n, p in self._params.items():
            setattr(mod, n, p.to(dtype))
        for n, m in self._mods.items():
            setattr(mod, n, m.to(dtype))
        return mod

    def half(self):
        return self.to(types.half)

    def float(self):
        return self.to(types.float)

    def double(self):
        return self.to(types.double)

    def train(self):
        return self.mutated(_training=True)

    def eval(self):
        return self.mutated(_training=False)

    def mutate(self, **attrs):
        return mutmod(self, **attrs)

    def mutated(self, **attrs):
        mod = self.copy()
        return mutmod(mod, **attrs)

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)

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
            strs.append(" (")
        strs.append("\n")
        for n, m in self._mods.items():
            strs.append(f"{' ' * pad}({n}): ")
            strs.extend(m.repr(pad + 3))
        if hasmods:
            strs.append(f"{' ' * (pad - 3)})\n")
        return "".join(strs)

    def xrepr(self) -> str:
        return self.__class__.__name__


def mutmod(mod: Module, **attrs):
    validattrs = {
        "mods": "_mods",
        "params": "_params",
        "training": "_training",
        "dtype": "_dtype",
    }
    for k, v in attrs.items():
        assert k in validattrs
        setattr(mod, k, v)
    return mod
