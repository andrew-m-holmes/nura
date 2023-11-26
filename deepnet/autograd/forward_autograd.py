import deepnet


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


def make_dual(tensor, tangent=None):
    if tangent is None:
        tangent = deepnet.zeros_like(tensor)
    return DualTensor(tensor, tangent)


def unpack_dual(dual_tensor):
    return dual_tensor.primal, dual_tensor.tangnet


def _pass_for_forward_autograd(context, output, *dual_tensors):
    if any(dual_tensor.use_grad for dual_tensor in dual_tensors):
        tangents = tuple(dual_tensor.tangent for dual_tensor in dual_tensors)
        tangent = context.apply(*tangents)
        output._set_grad_state(use_grad=True, grad_fn=None, is_leaf=False)
        return make_dual(output, tangent)
