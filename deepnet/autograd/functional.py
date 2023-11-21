import deepnet
from collections import deque
from deepnet.autograd.graph import AccumulateGrad, Node


def jacobian(input, func):
    # literally computes the full jacobian matrix for func
    # which holds the partial derivatives of the func outputs wrt
    # to every single input to that func
    pass


def vjp(inputs, vector, func, keep_graph=False):
    inputs = _vjp_setup_inputs(inputs, keep_graph)
    with deepnet.use_grad():
        output = func(*inputs)

    node = output.grad_fn
    queue = deque([(node, vector)])
    cotangents = []
    while queue:
        node, grad = queue.pop()
        next_nodes, grads = _vjp_next_grads_helper(node, grad)
        queue.extend(next_nodes)
        cotangents.extend(reversed(grads))

    output, cotangents = _vjp_post_process(output, cotangents, keep_graph)
    return output, cotangents


def _vjp_next_grads_helper(node, grad):
    next_grads, vectors, = [], []
    grads = _node_hijack_apply(node, grad)
    for node, grad, in zip(node.next_functions, grads):
        if node is not None and not isinstance(node.context, AccumulateGrad):
            next_grads.append((node, grad))
        elif node is not None and isinstance(node.context, AccumulateGrad):
            vectors.append(grad)
    return next_grads, vectors


def _vjp_post_process(output, vectors, keep_graph):
    if not keep_graph:
        output = output.clone().detach()
    vectors = tuple(reversed(vectors)) if len(vectors) > 1 else vectors[0]
    return output, vectors


def _vjp_setup_inputs(inputs, keep_graph):
    temp = inputs
    inputs = []
    for input in temp:
        if not keep_graph:
            input = input.clone().detach()
        input.use_grad = True
        inputs.append(input)
    return inputs


def jvp(input, vector, func):
    # evaluates the function at a particular input
    # (in forward mode) and returns the dot product
    # between the vector of interest and the computed
    # jacobian from the input
    pass


def grad(inputs, outputs):
    # will take the outputs and find the
    # gradients for every input passed
    # will not accumulate them
    pass


def _node_hijack_apply(node, grad):
    backward_fn = node.context.apply
    grad = backward_fn(grad)
    return grad


def _create_reverse_graph(output, root_descending=False):
    stack = [output.grad_fn]
    graph_nodes = []
    while stack:
        node = stack.pop()
        if node is not None and not isinstance(node.context, AccumulateGrad):
            graph_nodes.append(node)
            for child_node in node.next_functions:
                stack.append(child_node)
    if root_descending:
        graph_nodes.reverse()
    return graph_nodes
