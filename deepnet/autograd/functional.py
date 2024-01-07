import deepnet
import numpy as np
from types import FunctionType


def jacobian(input, func):
    # literally computes the full jacobian matrix for func
    # which holds the partial derivatives of the func outputs wrt
    # to every single input to that func
    pass


def vjp(primals, cotangent, func, *func_args, use_graph=False):
    _vjp_args_check(primals, cotangent, func, use_graph)
    primals, cotangent = _vjp_preprocess(
        primals, cotangent, use_graph)
    vjp_map = {primal: None for primal in primals}

    with deepnet.use_grad():
        output = func(*primals, *func_args)
    stack = [(output.grad_fn, cotangent)]
    while stack:
        node, cotangent = stack.pop()
        if _is_leaf_node(node) and node.context.tensor in vjp_map:
            if vjp_map[node.context.tensor] is None:
                vjp_map[node.context.tensor] = cotangent
            else:
                vjp_map[node.context.tensor] = _accumulate_grad(
                    vjp_map[node.context.tensor], cotangent)
        elif _is_intermediate_node(node):
            next_nodes, next_cotangents = _process_node(
                node, cotangent)
            for node, cotangent in zip(next_nodes, next_cotangents):
                stack.append((node, cotangent))
    return _vjp_post_process(vjp_map, output, use_graph)


def _vjp_post_process(vjp_map, output, use_graph):
    for primal, cotangent in vjp_map.items():
        if primal.dim() != cotangent.dim():
            cotangent = _reduce_sum_grad(primal, cotangent)
            cotangent = _broadcast_to_match(primal, cotangent)
            vjp_map[primal] = cotangent
    if not use_graph:
        del output.grad_fn
        output._set_grad_state(
            use_grad=False, grad_fn=None, is_leaf=True)
    return output, tuple(vjp_map.values())


def _vjp_preprocess(primals, cotangent, use_graph):
    tmp = primals
    primals = []
    for primal in tmp:
        if not use_graph:
            primal = primal.clone().detach()
        primal._set_grad_state(
            use_grad=True, grad_fn=None, is_leaf=True)
        primals.append(primal)
    cotangent._set_grad_state(
        use_grad=True, grad_fn=None, is_leaf=True)
    return primals, cotangent


def _vjp_args_check(primals, cotangent, func, use_graph):
    assert deepnet.is_all_tensor(*primals)
    assert deepnet.is_tensor(cotangent)
    assert isinstance(func, FunctionType)
    assert deepnet.is_py_bool(use_graph)
    assert all(tensor.dtype.differentiable() for tensor in primals)
    assert cotangent.dtype.differentiable()



def jvp(primals, tangents, func, *func_args, use_graph=False):
    _jvp_args_check(primals, tangents, func, use_graph)
    primals = _jvp_preprocess_primals(primals, tangents, use_graph)
    with deepnet.forward_ad(), deepnet.set_grad(use_graph):
        output = func(*primals, *func_args)
    output, tangent = _jvp_post_process(primals, output)
    return output, tangent


def _jvp_post_process(primals, output):
    for primal in primals:
        tensor, tangent = primal.undual(inplace=True)
    output, output_tangent = output.undual(inplace=True)
    return output, output_tangent


def _jvp_preprocess_primals(primals, tangents, use_graph):
    tmp = primals
    primals = []
    for primal, tangent in zip(tmp, tangents):
        if not use_graph:
            primal = primal.clone().detach()
        primal._set_grad_state(use_grad=use_graph,grad_fn=None,is_leaf=True)
        tangent._set_grad_state(use_grad=False,grad_fn=None,is_leaf=True)
        primal._set_dual_state(tangent, in_dual=True)
        primals.append(primal)
    return primals


def _jvp_args_check(primals, tangents, func, use_graph):
    assert deepnet.is_all_tensor(*primals)
    assert deepnet.is_all_tensor(*tangents)
    assert len(primals) == len(tangents)
    assert all(primal.dim() == tangent.dim() for primal, tangent in zip(primals, tangents))
    assert isinstance(func, FunctionType)
    assert deepnet.is_py_bool(use_graph)
    assert all(tensor.dtype.differentiable() for tensor in primals)
    assert all(tangent.dtype.differentiable() for tangent in tangents)


def grad(inputs, output, output_grad=None):
    _grad_args_check(inputs, output, output_grad)
    if deepnet.is_tensor(inputs):
        inputs = (inputs,)
    if output_grad is None:
        output_grad = deepnet.ones_like(output)
    grad_map = {tensor: deepnet.zeros_like(tensor) for tensor in inputs}
    stack = [(output.grad_fn, output_grad)]

    while stack:
        node, curr_grad = stack.pop()
        if _is_leaf_node(node) and node.context.tensor in grad_map:
            tensor = node.context.tensor
            if tensor.dim() != curr_grad.dim():
                curr_grad = _reduce_sum_grad(tensor, curr_grad)
            curr_grad = _broadcast_to_match(tensor, curr_grad)
            grad_map[tensor] = _accumulate_grad(grad_map[tensor], curr_grad)
        elif _is_intermediate_node(node):
            next_nodes, next_grads = _process_node(node, curr_grad)
            for next_node, next_grad in zip(next_nodes, next_grads):
                stack.append((next_node, next_grad))
    return tuple(grad_map.values())


def _grad_args_check(inputs, output, output_grad):
    assert deepnet.is_all_tensor(*inputs)
    assert deepnet.is_tensor(output)
    assert all(tensor.dtype.differentiable() for tensor in inputs) 
    assert output.dtype.differentiable()
    if output.nelem() > 1:
        assert output_grad is not None
        assert deepnet.is_tensor(output_grad)
        assert output_grad.dim() == output.dim()
        assert output_grad.dtype.differentiable()
    assert all(tensor.use_grad for tensor in inputs)
    assert output.use_grad
    assert output.grad_fn is not None

def _process_node(node, grad):
    next_grads = node.context.apply(grad)
    return node.next_functions, next_grads


def _is_leaf_node(node):
    if node is not None:
        return repr(node) == "AccumulateGrad"
    return False


def _is_intermediate_node(node):
    if node is not None:
        return "Backward" in repr(node)
    return False

def _accumulate_grad(grad_0, grad_1):
    grad_0.data += grad_1.data
    return grad_0

def _reduce_sum_grad(tensor, grad):
    padded_dim = np.pad(tensor.dim(), pad_width=(grad.ndim() - tensor.ndim(), 0), constant_values=0)
    mask = padded_dim != np.array(grad.dim())
    dims = tuple(i for i, bool_ in enumerate(mask) if bool_)
    keepdims = tensor.ndim() == grad.ndim()
    with deepnet.no_grad():
        grad = deepnet.sum(grad, dims, keepdims)
    return grad


def _broadcast_to_match(tensor, grad):
    if tensor.dim() != grad.dim():
        grad = deepnet.tensor(
            np.broadcast_to(grad.data, tensor.dim()))
    return grad

