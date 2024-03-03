import numpy as np
import neuro
from neuro.tensors import Tensor
from typing import Iterator, Optional, OrderedDict, Tuple, Any, Type, Union
from neuro.types import dtype
from collections import OrderedDict
from copy import copy
from numpy import ndarray


class Parameter(Tensor):

    def __init__(
        self,
        data: Optional[Union[Tensor, ndarray]] = None,
        usesgrad=True,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:

        if data is None:
            data = np.empty(0)
        if isinstance(data, Tensor):
            data = data.data
        if dtype is None:
            dtype = neuro.float
        super().__init__(
            data, usegrad=usesgrad, grad=None, backfn=None, leaf=True, _dtype=dtype
        )

    def to(self, dtype: Type[dtype]):
        return Parameter(self.data, self.usegrad, dtype)

    def half(self):
        return Parameter(self.data, self.usegrad, neuro.half)

    def float(self):
        return Parameter(self.data, self.usegrad, neuro.float)

    def double(self):
        return Parameter(self.data, self.usegrad, neuro.double)

    def __repr__(self) -> str:
        return super().__repr__().replace("tensor", "param")


class Buffer(Tensor):

    def __init__(
        self,
        data: Optional[Union[Tensor, ndarray]] = None,
        dtype: Optional[Type[dtype]] = None,
    ) -> None:
        if data is None:
            data = np.empty(0)
        if isinstance(data, Tensor):
            data = data.data
        if dtype is None:
            dtype = neuro.float
        super().__init__(
            data, usegrad=False, grad=None, backfn=None, leaf=True, _dtype=dtype
        )

    def to(self, dtype: Type[dtype]):
        return Buffer(self.data, dtype)

    def half(self):
        return Buffer(self.data, neuro.half)

    def float(self):
        return Buffer(self.data, neuro.float)

    def double(self):
        return Buffer(self.data, neuro.double)

    def __repr__(self) -> str:
        return super().__repr__().replace("tensor", "buff")


class Module:

    def __init__(self, *args, **kwargs) -> None:
        self._mods: OrderedDict[str, "Module"] = OrderedDict()
        self._params: OrderedDict[str, Parameter] = OrderedDict()
        self._buffs: OrderedDict[str, Buffer] = OrderedDict()
        self._training: bool = True

    @property
    def mods(self) -> OrderedDict[str, "Module"]:
        return self._mods

    @property
    def params(self) -> OrderedDict[str, Parameter]:
        return self._params

    @property
    def buffs(self) -> OrderedDict[str, Buffer]:
        return self._buffs

    @property
    def training(self) -> bool:
        return self._training

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def forward(self):
        raise NotImplemented

    def itermods(self, s="") -> Iterator[Tuple[str, "Module"]]:
        if not s:
            s = self.name().lower()
        yield s, self
        for n, m in self.mods.items():
            yield from m.itermods(f"{s}.{n}")

    def iterparams(self) -> Iterator[Tuple[str, Parameter]]:
        yield from iter(self.params.items())
        for m in self.mods.values():
            yield from m.iterparams()

    def iterbuffs(self) -> Iterator[Tuple[str, Buffer]]:
        yield from iter(self.buffs.items())
        for m in self.mods.values():
            yield from m.iterbuffs()

    def hasmods(self) -> bool:
        return len(self.mods) > 0

    def hasparams(self) -> bool:
        return len(self.params) > 0

    def hasbuffs(self) -> bool:
        return len(self.buffs) > 0

    def train(self) -> "Module":
        self._training = True
        return self

    def eval(self) -> "Module":
        self._training = False
        return self

    def to(self, dtype: Type[dtype]):
        params = map(lambda item: (item[0], item[1].to(dtype)), self.params.items())
        self._params = OrderedDict(params)
        buffs = map(lambda item: (item[0], item[1].to(dtype)), self.buffs.items())
        self._buffs = OrderedDict(buffs)
        for m in self.mods.values():
            m.half()
        return self

    def half(self) -> "Module":
        return self.to(neuro.half)

    def float(self) -> "Module":
        return self.to(neuro.float)

    def double(self) -> "Module":
        return self.to(neuro.double)

    def mutate(self, **attrs) -> "Module":
        return mutmodule(self, **attrs)

    def mutated(self, **attrs) -> "Module":
        mod = copy(self)
        return mutmodule(mod, **attrs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._mods[name] = value
        elif isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Buffer):
            self._buffs[name] = value
        self.__dict__[name] = value

    def __repr__(self):
        return self.name()

    def display(self, n=1) -> str:
        strs = [repr(self)]
        if n == 1:
            strs.append(": [ \n")
        else:
            strs.append("\n")
        for i, m in enumerate(self.mods.values()):
            strs.extend(f"{'   ' * n}[{i}]: {m.display(n + 1)}")
        if n == 1:
            strs.append("]")
        return "".join(strs)


def mutmodule(module: Module, **attrs: Any) -> Module:
    validattrs = {
        "mods": "_mods",
        "params": "_params",
        "buffs": "_params",
        "training": "_training",
    }
    for name, val in attrs.items():
        if name in validattrs:
            setattr(module, validattrs[name], val)
    return module
