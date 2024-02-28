from typing import Optional
from neuro.tensors import Tensor
from neuro.utils import empty


class Parameter:

    def __init__(self, tensor: Optional[Tensor] = None, keepsgrad=True) -> None:
        self._tensor: Tensor = (
            empty(0) if tensor is None else tensor.mutated(usegrad=keepsgrad)
        )
        self._keepsgrad: bool = keepsgrad

    @property
    def tensor(self):
        return self._tensor

    @property
    def keepsgrad(self):
        return self._keepsgrad

    def __repr__(self) -> str:
        return f"param: {self.tensor}"
