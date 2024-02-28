from typing import Iterator, Optional, Dict, Tuple, Any, Union
from deepnet.nn.parameter import Parameter
from deepnet.tensors import Tensor


class Module:

    def __init__(self, *args, **kwargs) -> None:
        self._mods: Dict[str, Optional["Module"]] = {}
        self._params: Dict[str, Optional[Parameter]] = {}
        self._buffs: Dict[str, Optional[Tensor]]
        self._active: bool = True
        self._args: Tuple[Any, ...] = args
        self._kwargs: Dict[Any, Any] = kwargs

    @property
    def active(self):
        return self._active

    def forward(self, *args, **kwargs):
        raise NotImplemented

    def params(self) -> Iterator[Tuple[str, Optional[Parameter]]]:
        return iter((n, p) for n, p in self._params.items())

    def mods(self) -> Iterator[Tuple[str, Optional["Module"]]]:
        return iter((n, m) for n, m in self._mods.items())

    def allmods(self):
        pass

    def train(self):
        self._trainable = True
        return self

    def eval(self):
        self._trainable = False
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params[name] = value
            self.__dict__[name] = value
        elif isinstance(value, Module):
            self._mods[name] = value
            self.__dict__[name] = value
        else:
            names = {"_mods", "_params", "_buffs", "_active", "_args", "_kwargs"}
            assert name in names, f"invalid attr: {name}"
            self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.__class__.__name__
