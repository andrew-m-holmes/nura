import deepnet
import numpy as np
from deepnet import Tensor
from typing import Tuple


class DualTensor:

    def __init__(self, primal, tangent) -> None:
        self.primal = primal
        self.tangent = tangent

    @property
    def data(self):
        return self.primal.data

    @property
    def use_grad(self):
        return self.primal.use_grad

    def __repr__(self):
        return f"dual_tensor(primal: {repr(self.primal)}, tangent: {repr(self.tangent)})"


def make_dual(tensor, tangent=None) -> DualTensor:
    if tangent is None:
        tangent = deepnet.zeros_like(tensor)
    assert isinstance(tensor, Tensor), \
        f"Invalid argument: {tensor}, only Tensors can be made dual"
    assert isinstance(tangent, Tensor), \
        f"Invalid argument: {tangent}, tangent must be a Tensor"
    assert _is_differentiable(tensor, tangent), \
        "Can only differentiate Tensors of float dtypes"
    return DualTensor(tensor, tangent)


def unpack_dual(dual_tensor) -> Tuple[Tensor, Tensor]:
    return dual_tensor.primal, dual_tensor.tangnet


def _pass_for_forward_autograd(context, output, *dual_tensors):
    tangents = tuple(dual_tensor.tangent for dual_tensor in dual_tensors)
    tangent = context.apply(*tangents)
    output._set_grad_state(use_grad=True, grad_fn=None, is_leaf=False)
    return make_dual(output, tangent)


def _is_differentiable(*tensors):
    dtypes = [float, np.float16, np.float32, np.float64, np.float128]
    return all(tensor.dtype() in dtypes for tensor in tensors)
