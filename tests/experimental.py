import numpy as np
import jax.numpy as jnp
import deepnet
import deepnet.autograd as deepnet_autograd
from deepnet.autograd.functional import vjp, jvp, grad, jacfwd, _get_perturbations


def main():

   a = deepnet.rand((4,)) 
   b = deepnet.rand((4,))
   out, jac = jacfwd((a, b), deepnet.mul, use_graph=True)
   out.backward(deepnet.ones_like(out))
   print(jac)
   print(a.grad)


if __name__ == "__main__":
    main()
