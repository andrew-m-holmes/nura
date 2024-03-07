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
        self._args = args
        self._kwargs = kwargs

    @property
    def training(self) -> bool:
        return self._training

    def forward(self):
        raise NotImplemented

    def modules(self, s="") -> Iterator[Tuple[str, "Module"]]:
        if not s:
            s = self.__class__.__name__.lower()
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
        strs = [self.__class__.__name__, "\n"]
        for n, m in self._modules.items():
            strs.append(f"{n}: ")
            strs.extend(m.xrepr())
        return "".join(strs)
