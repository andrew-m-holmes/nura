import neuro.functions as function
from neuro.tensors import Tensor
from typing import Optional


def linear(inpt: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    out = function.Matmul.apply(inpt, weight.transpose())
    if bias is not None:
        out = out + bias
    return out
