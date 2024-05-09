import nura.types as types
from nura.types import dtype
from nura.nn.parameter import Parameter
from collections import OrderedDict
from typing import Type, Iterator, Tuple, Any
from copy import copy


class Module:

    def __init__(self) -> None:
        self._modules: OrderedDict[str, "Module"] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._training: bool = True

    @property
    def training(self) -> bool:
        return self._training

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplemented

    def modules(self) -> Iterator["Module"]:
        yield from self._modules.values()
        for m in self._modules.values():
            yield from m.modules()

    def namedmodules(self) -> Iterator[Tuple[str, "Module"]]:
        yield from self._modules.items()
        for m in self._modules.values():
            yield from m.namedmodules()

    def parameters(self) -> Iterator[Parameter]:
        yield from self._parameters.values()
        for m in self._modules.values():
            yield from m.parameters()

    def namedparameters(self) -> Iterator[Tuple[str, Parameter]]:
        yield from self._parameters.items()
        for m in self._modules.values():
            yield from m.namedparameters()

    def to(self, dtype: Type[dtype]) -> "Module":
        parameters = OrderedDict((n, p.to(dtype)) for n, p in self._parameters.items())
        modules = OrderedDict((n, m.to(dtype)) for n, m in self._modules.items())
        mod = copy(self)
        mod._parameters = parameters
        mod._modules = modules
        return mod

    def half(self) -> "Module":
        return self.to(types.half)

    def float(self) -> "Module":
        return self.to(types.float)

    def double(self) -> "Module":
        return self.to(types.double)

    def train(self):
        self._training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        self._training = False
        for m in self._modules.values():
            m.eval()

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        if isinstance(value, Parameter):
            self._parameters[name] = value
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.reprhelp()[:-1]

    def reprhelp(self, pad=3) -> str:
        strs = [self.xrepr()]
        if hasmodules := len(self._modules):
            strs.append(": (")
        strs.append("\n")
        for n, m in self._modules.items():
            strs.append(f"{' ' * pad}({n}): ")
            strs.extend(m.reprhelp(pad + 3))
        if hasmodules:
            strs.append(f"{' ' * (pad - 3)})\n")
        return "".join(strs)

    def xrepr(self) -> str:
        return f"{self.__class__.__name__}()"
