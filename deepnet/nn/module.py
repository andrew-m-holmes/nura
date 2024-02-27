from typing import Optional, Dict, Tuple, Any
from deepnet.nn.parameter import Parameter


class Module:

    def __init__(self, *args, **kwargs) -> None:
        self._params: Dict[str, Optional[Parameter]] = {}
        self._trainable: bool = True
        self._args: Tuple[Any, ...] = args
        self._kwargs: Dict[Any, Any] = kwargs

    @property
    def trainable(self):
        return self._trainable

    def forward(self, *args, **kwargs):
        raise NotImplemented

    def params(self):
        return self._params

    def train(self):
        self._trainable = True
        return self

    def infer(self):
        self._trainable = False
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params[name] = value
            self.__dict__[name] = value
        else:
            validnames = {"_trainable", "_params", "_args", "_kwargs"}
            assert name in validnames
            self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.__class__.__name__.lower()
