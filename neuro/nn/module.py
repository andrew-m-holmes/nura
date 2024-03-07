from neuro.nn.parameter import Parameter
from neuro.tensors import Tensor
from collections import OrderedDict


class Module:

    def __init__(self) -> None:
        self._modules: OrderedDict[str, "Module"] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Tensor] = OrderedDict()
        self._training: bool = True

    @property
    def training(self) -> bool:
        return self._training

    def forward(self):
        raise NotImplemented

    def train(self):
        self._training = True
        return self

    def eval(self):
        self._training = False
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.xrepr()

    def xrepr(self) -> str:
        strs = [self.__class__.__name__, "\n"]
        for n, m in self._modules.items():
            strs.append(f"{n}: ")
            strs.extend(m.xrepr())
        return "".join(strs)
