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


def make_dual(tensor, tangent=None):
    if tangent is None:
        tangent = deepnet.zeros_like(tensor)
    return DualTensor(tensor, tangent)


def unpack_dual(dual_tensor):
    return dual_tensor.primal, dual_tensor.tangnet
