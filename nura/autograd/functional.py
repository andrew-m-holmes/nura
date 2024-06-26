import numpy as np
import nura
from nura.tensors import Tensor
from nura.autograd.graph import Node, toposort
from typing import Any, Dict, Generator, Tuple, Optional, Callable, Union, Set
from collections import deque


def backward(
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    input: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> None:

    output, grad, input = _tupify(output), _tupify(grad), _tupify(input)
    if len(output) != len(set(output)):
        raise ValueError("Cannot run backward, received duplicate output tensors")
    if not all(t.gradfn is not None and t.gradtensor for t in output):
        raise ValueError(
            "Cannot run backward, all tensors must be on computational graph"
        )
    if any(output[i].nelem > 1 and i >= len(grad) for i in range(len(output))):
        raise ValueError(
            "Cannot run backward, gradient must be supplied for outputs that are not scalars"
        )
    if len(grad) != len(set(grad)):
        raise ValueError("Cannot run backward, received duplicate gradient tensors")
    if len(grad) > len(output):
        raise ValueError(
            "Cannot run backward, received more gradients than output tensors"
        )
    if not all(grad[i].dtype is output[i].dtype for i in range(len(grad))):
        raise ValueError(
            "Cannot run backward, received one or more gradients with different types than output tensors"
        )
    if not all(grad[i].dim == output[i].dim for i in range(len(grad))):
        raise ValueError(
            "Cannot run backward, received one or more gradients with different dimensions than output tensors"
        )
    if len(input) != len(set(input)):
        raise ValueError("Cannot run backward, received duplicate input tensors")
    if not all(i.gradtensor and i.usegrad and i.gradfn is not None for i in input):
        raise ValueError(
            "Cannot run backward, received inputs not on computational graph"
        )
    _backward(output, grad, input)


def _backward(
    output: Tuple[Tensor, ...], grad: Tuple[Tensor, ...], input: Tuple[Tensor, ...]
) -> None:

    nodes = tuple(o.gradfn for o in output if o.gradfn is not None)
    topotuple = toposort(nodes)
    retain = _getretain(topotuple, input)
    gradmap = _getgradmap(nodes, output, grad)
    queue = deque(topotuple)

    while queue:
        node = queue.popleft()
        nodegrad = gradmap[node]
        if node in retain:
            _accumulate(node, nodegrad)
        if node.edges:
            gradoutput = _tupify(node.apply(nodegrad))
            for edge, edgegrad in zip(node.edges, gradoutput):
                if edge is None:
                    continue
                if edge not in gradmap:
                    gradmap[edge] = nura.zeroslike(edge.output)
                if _mismatch(edge.output, edgegrad):
                    edgegrad = _sumgrad(edge.output, edgegrad)
                gradmap[edge] += edgegrad
        gradmap.pop(node)


def grad(
    input: Union[Tuple[Tensor, ...], Tensor],
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> Tuple[Tensor, ...]:

    output, grad, input = _tupify(output), _tupify(grad), _tupify(input)
    if len(output) != len(set(output)):
        raise ValueError("Cannot run backward, received duplicate output tensors")
    if not all(t.gradfn is not None and t.gradtensor for t in output):
        raise ValueError("Cannot run grad, all tensors must be on computational graph")
    if any(output[i].nelem > 1 and i >= len(grad) for i in range(len(output))):
        raise ValueError(
            "Cannot run grad, gradient must be supplied for outputs that are not scalars"
        )
    if len(grad) != len(set(grad)):
        raise ValueError("Cannot run backward, received duplicate gradient tensors")
    if len(grad) > len(output):
        raise ValueError("Cannot run grad, received more gradients than output tensors")
    if not all(grad[i].dtype is output[i].dtype for i in range(len(grad))):
        raise ValueError(
            "Cannot run grad, received one or more gradients with different types than output tensors"
        )
    if not all(grad[i].dim == output[i].dim for i in range(len(grad))):
        raise ValueError(
            "Cannot run grad, received one or more gradients with different dimensions than output tensors"
        )
    if len(input) != len(set(input)):
        raise ValueError("Cannot run backward, received duplicate input tensors")
    if not all(i.gradtensor and i.usegrad and i.gradfn is not None for i in input):
        raise ValueError("Cannot run grad, received inputs not on computational graph")
    return _grad(input, output, grad)


def _grad(
    input: Tuple[Tensor, ...], output: Tuple[Tensor, ...], grad: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    nodes = tuple(o.gradfn for o in output if o.gradfn is not None)
    topotuple = toposort(nodes)
    retain = _getretain((), input)
    gradmap = _getgradmap(nodes, output, grad)
    queue = deque(topotuple)

    while queue:
        node = queue.popleft()
        nodegrad = gradmap[node]
        if node.edges:
            gradoutput = _tupify(node.apply(nodegrad))
            for edge, edgegrad in zip(node.edges, gradoutput):
                if edge is None:
                    continue
                if edge not in gradmap:
                    gradmap[edge] = nura.zeroslike(edge.output)
                if _mismatch(edge.output, edgegrad):
                    edgegrad = _sumgrad(edge.output, edgegrad)
                gradmap[edge] += edgegrad
        if node not in retain:
            gradmap.pop(node)
    return tuple(gradmap[i.gradfn] for i in input if i.gradfn is not None)


def _getretain(node: Tuple[Node, ...], input: Tuple[Tensor, ...]) -> Set[Node]:
    retain = set()
    for n in node:
        if n.accumulate:
            retain.add(n)
    for i in input:
        if i.gradfn is not None:
            retain.add(i.gradfn)
    return retain


def _getgradmap(
    node: Tuple[Node, ...], output: Tuple[Tensor, ...], grad: Tuple[Tensor, ...]
) -> Dict[Node, Tensor]:
    gradmap = {}
    for i, (n, o) in enumerate(zip(node, output)):
        if i < len(grad):
            gradmap[n] = grad[i]
        else:
            gradmap[n] = nura.oneslike(o)
    return gradmap


def _tupify(input: Optional[Union[Tuple[Tensor, ...], Tensor]]) -> Tuple[Tensor, ...]:
    if input is None:
        return ()
    if isinstance(input, Tensor):
        return (input,)
    return input


def _mismatch(tensor: Tensor, grad: Tensor) -> bool:
    return tensor.dim != grad.dim


def _sumgrad(tensor: Tensor, grad: Tensor) -> Tensor:
    if tensor.ndim <= grad.ndim:
        pad = np.pad(tensor.dim, (grad.ndim - tensor.ndim, 0), constant_values=0)
        mismatched = pad != grad.dim
        dim = np.nonzero(mismatched)[0]
        grad = grad.sum(dim=tuple(dim), keepdims=True)
        if tensor.ndim < grad.ndim:
            dim = tuple(range(grad.ndim - tensor.ndim))
            grad = grad.squeeze(dim)
    else:
        grad = nura.zeroslike(tensor) + grad
    return grad


def _accumulate(node: Node, grad: Tensor) -> None:
    if node.output.dim != grad.dim:
        raise ValueError(
            f"Cannot accumulate gradient, node output dimensions does not match gradient dimensions ({node.output.dim} != {grad.dim})"
        )
    if node.output.dtype != grad.dtype:
        raise ValueError(
            f"Cannot accumulate gradient, node output type does not match gradient type ({node.output.dtype.name()} != {grad.dtype.name()})"
        )
    if node.output._grad is None:
        node.output._grad = nura.zeroslike(node.output)
    node.output._grad += grad


def vjp(
    input: Union[Tuple[Tensor, ...], Tensor],
    vector: Tensor,
    func: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:

    input = _tupify(input)
    if not all(i.gradtensor for i in input):
        raise ValueError(
            "Cannot run vector-jacobian product, all input tensors must be differentiable"
        )
    if not vector.gradtensor:
        raise ValueError(
            "Cannot run vector-jacobian product, vector must be differentiable"
        )

    input = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in input)
    vector = vector.mutated(usegrad=False, grad=None)
    output, vjps = _vjp(input, vector, func, *args, **kwargs)
    return output.mutated(usegrad=False, gradfn=None, leaf=True), vjps


def _vjp(
    input: Tuple[Tensor, ...],
    vector: Tensor,
    func: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    with nura.usegrad():
        output = func(*input, *args, **kwargs)
    vjps = _grad(input, (output,), (vector,))
    return output, vjps


def jvp(
    input: Union[Tuple[Tensor, ...], Tensor],
    vector: Union[Tuple[Tensor, ...], Tensor],
    func: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    input = _tupify(input)
    vector = _tupify(vector)
    if not all(i.gradtensor for i in input):
        raise ValueError(
            "Cannot run jacobian-vector product, all input tensors must be differentiable"
        )
    if len(input) != len(vector):
        raise ValueError(
            "Cannot run jacobian-vector product, the amount of vectors supplied must match the amount of input tensors"
        )
    if not all(v.gradtensor for v in vector):
        raise ValueError(
            "Cannot run jacobian-vector product, vector must be differentiable"
        )
    if any(v.dim != i.dim for i, v in zip(input, vector)):
        raise ValueError(
            "Cannot run jacobian-vector product, vector(s) must share dimensions of corresponding input tensors"
        )

    input = tuple(t.mutated(usegrad=True, grad=v) for t, v in zip(input, vector))
    output, grad = _jvp(input, func, *args, **kwargs)
    return output.mutated(usegrad=False, leaf=True), grad


def _jvp(
    input: Tuple[Tensor, ...],
    func: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    with nura.forwardmode():
        output = func(*input, *args, **kwargs)
        assert output.grad is not None
        grad = output.grad
    return output, grad


def jacrev(
    input: Union[Tuple[Tensor, ...], Tensor],
    func: Callable[..., Tensor],
    pos: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:

    input = _tupify(input)
    if not all(t.gradtensor for t in input):
        raise ValueError(
            "Cannot compute reverse-ad jacobian, input tensor(s) must be differentiable"
        )
    if pos not in range(-len(input), len(input)):
        raise ValueError(
            "Cannot compute reverse-ad jacobian, selected position is out of range"
        )

    input = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in input)
    with nura.usegrad():
        output = func(*input, *args, **kwargs)
    tensor = input[pos]
    jac = _getjac(tensor, output)
    perts = _getperts(output)

    for row, pert in zip(np.ndindex(output.dim), perts):
        _, vjps = _vjp(input, pert, func, *args, **kwargs)
        jacrow = vjps[pos]
        slc = row + (...,)
        jac[slc] = jacrow
    return output, jac


def jacfwd(
    input: Union[Tuple[Tensor, ...], Tensor],
    func: Callable[..., Tensor],
    pos: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:

    input = _tupify(input)
    if not all(t.gradtensor for t in input):
        raise ValueError(
            "Cannot compute forward-ad jacobian, input tensor(s) must be differentiable"
        )
    if pos not in range(-len(input), len(input)):
        raise ValueError(
            "Cannot compute forward-ad jacobian, selected position is out of range"
        )

    with nura.forwardmode():
        output = func(*input, *args, **kwargs)
    tensor = input[pos]
    perts = _getperts(tensor)
    jac = _getjac(tensor, output)
    colinput = [
        t.mutated(usegrad=True, grad=nura.zeroslike(t)) if i != pos else t
        for i, t in enumerate(input)
    ]

    for col, pert in zip(np.ndindex(tensor.dim), perts):
        colinput[pos] = colinput[pos].mutated(usegrad=True, grad=pert)
        _, jaccol = _jvp(tuple(colinput), func, *args, **kwargs)
        slc = (...,) + col
        jac[slc] = jaccol
    return output, jac


def _getperts(tensor: Tensor) -> Generator[Tensor, None, None]:
    nelem, dim, dtype = tensor.nelem, tensor.dim, tensor.dtype
    perts = nura.zeros((nelem,) + dim).to(dtype)
    arange = np.arange(nelem)
    indices = np.unravel_index(arange, dim)
    slc = (arange,) + indices
    perts[slc] = 1.0
    return (perts[i] for i in range(nelem))


def _getjac(tensor: Tensor, output: Tensor) -> Tensor:
    dim = output.dim + tensor.dim
    jac = nura.zeros(dim).to(output.dtype)
    return jac
