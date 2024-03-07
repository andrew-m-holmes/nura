import neuro
from neuro.nn.parameter import Parameter
from neuro.tensors import Tensor
from typing import Iterator, Tuple, Any, Type, Union
from neuro.types import dtype
from collections import OrderedDict
from copy import copy


class Module:

    def __init__(self, *args, **kwargs) -> None:
        self._modules: OrderedDict[str, "Module"] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Tensor] = OrderedDict()
        self._training: bool = True

    @property
    def training(self) -> bool:
        return self._training

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def forward(self):
        raise NotImplemented

    def to(self, dtype: Type[dtype]):
        for n, p in self.parameters():
            self._parameters[n] = p.to(dtype)
        for _, m in self.modules():
            m.to(dtype)
        return self

    def half(self):
        return self.to(neuro.half)

    def float(self):
        return self.to(neuro.float)

    def double(self):
        return self.to(neuro.double)

    def modules(self, s="") -> Iterator[Tuple[str, "Module"]]:
        if not s:
            s = self.name().lower()
        yield s, self
        for n, m in self._modules.items():
            yield from m.modules(f"{s}.{n}")

    def parameters(self) -> Iterator[Tuple[str, Parameter]]:
        yield from self._parameters.items()
        for m in self._modules.values():
            yield from m.parameters()

    def train(self) -> "Module":
        self._training = True
        return self

    def eval(self) -> "Module":
        self._training = False
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        if isinstance(value, Parameter):
            self._parameters[name] = value
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.xrepr()

    def xrepr(self) -> str:
        strs = [self.name(), "\n"]
        for n, m in self._modules.items():
            strs.append(f"{n}: ")
            strs.extend(m.xrepr())
        return "".join(strs)
