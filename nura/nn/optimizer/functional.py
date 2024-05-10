import nura
from nura.tensors import Tensor
from nura.nn.parameter import Parameter
from typing import Optional, Tuple


def computedecay(tensor: Tensor, grad: Tensor, decay: float) -> Tensor:
    return grad + tensor * decay


def sgd(
    parameter: Parameter,
    velocity: Tensor,
    learnrate: float,
    momentum: float = 0.9,
    nesterov: bool = False,
    decay: Optional[float] = None,
    graph: bool = False,
) -> Tensor:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")

    with nura.setgrad(graph):
        grad = (
            computedecay(parameter, parameter.grad, decay)
            if decay is not None
            else parameter.grad
        )

        if nesterov:
            grad += momentum * velocity
        update = momentum * velocity + learnrate * grad
        return update


def rmsprop(
    parameter: Parameter,
    velocity: Tensor,
    learnrate: float,
    alpha: float = 0.9,
    decay: Optional[float] = None,
    eps: float = 1e-8,
    graph: bool = False,
) -> Tuple[Tensor, Tensor]:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")

    with nura.setgrad(graph):
        grad = (
            computedecay(parameter, parameter.grad, decay)
            if decay is not None
            else parameter.grad
        )

        nextvel = alpha * velocity + (1 - alpha) * nura.square(grad)
        update = learnrate / nura.sqrt(nextvel + eps) * grad
        return update, nextvel


def adam(
    parameter: Parameter,
    velocities: Tuple[Tensor, Tensor],
    learnrate: float,
    timestep: int,
    betas: Tuple[float, float] = (0.9, 0.99),
    decay: Optional[float] = None,
    eps: float = 1e-8,
    graph: bool = False,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    if parameter.grad is None:
        raise ValueError("Cannot compute update gradient, parameter.grad is None")

    with nura.setgrad(graph):
        grad = (
            computedecay(parameter, parameter.grad, decay)
            if decay is not None
            else parameter.grad
        )

        mt = betas[0] * velocities[0] + (1 - betas[0]) * grad
        vt = betas[1] * velocities[1] + (1 - betas[1]) * nura.square(grad)
        mthat = 1 / (1 - betas[0] ** timestep) * mt
        vthat = 1 / (1 - betas[1] ** timestep) * vt
        update = mthat / nura.sqrt(vthat + eps) * learnrate
        return update, (mt, vt)
