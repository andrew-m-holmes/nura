import numpy as np
import deepnet
from deepnet import Tensor


def jacobian(input, func):
    # literally computes the full jacobian matrix for func
    # which holds the partial derivatives of the func outputs wrt
    # to every single input to that func
    pass


def vjp(primals, cotangent, func, use_graph=False):
    primals, cotangent = _vjp_pre_process(primals, cotangent, use_graph)
    with deepnet.use_grad():
        output = func(*primals)

    cotangents = []
    stack = [(output.grad_fn, cotangent)]
    while stack:
        node, cotangent = stack.pop()
        if _is_leaf_node(node):
            cotangents.append(cotangent)
        elif _is_intermediate_node(node):
            next_nodes, next_cotangents = _process_node(node, cotangent)
            for node, cotangent in zip(next_nodes, next_cotangents):
                if _is_leaf_node(node) or _is_intermediate_node(node):
                    stack.append((node, cotangent))
    return _vjp_post_process(output, cotangents, use_graph)


def _process_node(node, cotangent):
    next_cotangents = node.context.apply(cotangent)
    return node.next_functions, next_cotangents


def _vjp_post_process(output, cotangents, use_graph):
    if not use_graph:
        del output.grad_fn
        output._set_grad_state(False, None, False)
    return output, tuple(reversed(cotangents))


def _vjp_pre_process(primals, cotangent, use_graph):

    temp = primals
    primals = []
    for primal in temp:
        if not use_graph:
            primal = primal.detach().clone()
        primal.use_grad = True
        primals.append(primal)
    cotangent.use_grad = True
    return primals, cotangent


def _is_leaf_node(node):
    if node is not None:
        return hasattr(node.context, "tensor")
    return False


def _is_intermediate_node(node):
    if node is not None:
        return not hasattr(node.context, "tensor")


def jvp(primals, tangents, func):
    dual_tensors = _jvp_pre_process(primals, tangents)
    with deepnet.forward_autograd():
        output = func(*dual_tensors)
    return _jvp_post_process(output)


def _jvp_post_process(output):
    if isinstance(output, tuple):
        return tuple(dual_tensor.unpack() for dual_tensor in output)
    return output.unpack()


def _jvp_pre_process(primals, tangents):

    dual_tensors = [deepnet.dual_tensor(primal.detach().clone(
    ), tangent.detach().clone()) for primal, tangent in zip(primals, tangents)]
    return dual_tensors


def grad(inputs, outputs):
    # will take the outputs and find the
    # gradients for every input passed
    # will not accumulate them
    pass


def _is_differentiable(*tensors):
    dtypes = [float, np.float16, np.float32, np.float64, np.float128]
    return all(tensor.dtype in dtypes for tensor in tensors)
