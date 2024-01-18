import numpy as np
import jax.numpy as jnp
import deepnet
import deepnet.autograd as deepnet_autograd
from deepnet.autograd.functional import vjp, jvp, grad, jacfwd, _get_perturbations


def main():

    a = deepnet.rand(5, use_grad=True, dtype=deepnet.float)
    b = deepnet.rand(1, use_grad=True).float()
    def f(a, b):
        return a * b - b ** a
    
    out, jac= jacfwd((a, b), f, index=1, use_graph=True)
    out.backward(deepnet.ones_like(out))
    print(a.grad)
    print(b.grad)
    print(jac.sum())
    print(jac)

if __name__ == "__main__":
    main()
