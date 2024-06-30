import nura
from nura.tensors import Tensor
from nura.autograd.function import Function, Context
from nura.types import Scalar
from typing import Optional, Set, Union, Type

primals: Set[Tensor] = set()


def cleanup():
    for t in primals:
        t.cleargrad()


def primal(
    tensor: Union[Tensor, Scalar], grad: Optional[Union[Tensor, Scalar]] = None
) -> Tensor:
    if not nura.Autograd.forwardmode():
        raise RuntimeError("Cannot create primal, forward mode AD is disabled")
    if not isinstance(tensor, Tensor):
        tensor = nura.tensor(tensor)
    if grad is not None:
        if not isinstance(grad, Tensor):
            grad = nura.tensor(grad)
        if tensor.dtype is not grad.dtype:
            raise ValueError(
                f"Cannot create primal, type mismtach between tensor and grad ({tensor.dtype.name()} != {grad.dtype.name()})"
            )
        if tensor.dim != grad.dim:
            raise ValueError(
                f"Cannot create primal, dimension mismtach between tensor and grad ({tensor.dim} != {grad.dim})"
            )
    else:
        grad = nura.zeroslike(tensor)
    p = tensor.mutated(grad=grad)
    primals.add(p)
    return p


def primalify(output: Tensor, function: Type[Function], context: Context) -> None:
    direction = tuple(
        t.grad if t.grad is not None else nura.zeroslike(t) for t in context.tensors()
    )
    grad = nura.tensor(function.tangent(context, *direction))
    output.mutate(grad=grad)
    primals.add(output)
