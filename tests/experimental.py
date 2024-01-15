import numpy as np
import jax.numpy as jnp
import deepnet
import deepnet.autograd as deepnet_autograd
from deepnet.autograd.functional import vjp, jvp, grad, jac, _get_perturbations
from jax import jacfwd, jacrev, random


def main():

    a = deepnet.rand((3, 3), use_grad=True, dtype=deepnet.float)
    b = deepnet.rand((3, 3), use_grad=True).float()
    def f(a, b):
        return a * b
    
    jacobian = jac((a, b), f, index=1)
    print(jacobian)


    key = random.key(0)
    key_0, key_1 = random.split(key)
    a = random.normal(key_1, (3,))
    b  = random.normal(key_0, (1,))
    def f(a, b):
        return a * b

    jacobian = jacfwd(f)(a, b)


if __name__ == "__main__":
    main()
