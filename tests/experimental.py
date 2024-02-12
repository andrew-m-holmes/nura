import jax
import jax.numpy as jnp
import numpy as np
import deepnet as dn
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd

def main():

    a = np.random.rand(4)
    b = np.random.rand(4)
    c = np.random.rand(1)
    v = np.ones(4)
    u = np.ones(1)
   
    dn_a = dn.tensor(a, usegrad=True).float()
    dn_b = dn.tensor(b, usegrad=True).float()
    dn_c = dn.tensor(c, usegrad=True).float()
    dn_v = dn.tensor(v).float()
    dn_u = dn.tensor(u).float()

    def f(a, b, c):
        return a * b + c 
    
    primal = f(dn_a, dn_b, dn_c)
    cotangents = grad((dn_a, dn_b, dn_c), primal, dn_v)
    print(primal)
    print(cotangents)

    primal.backward(dn_v)
    print(dn_a.grad, dn_b.grad, dn_c.grad)

    primal, tangents = jax.jvp(f, (a, b, c), (v, v, u))
    print(primal)
    print(tangents)

    primal, tangents = jvp((dn_a, dn_b, dn_c), (dn_v, dn_v, dn_u), f)
    print(primal)
    print(tangents)

    primal, jax_vjp_f = jax.vjp(f, a, b, c)
    cotangents = jax_vjp_f(v)
    print(primal)
    print(cotangents)

    primal, cotangents = vjp((dn_a, dn_b, dn_c), dn_v, f)
    print(primal)
    print(cotangents)

    primal, jac = jacfwd((dn_a, dn_b, dn_c), f, 2)
    print(primal)
    print(jac)



if __name__ == "__main__":
    main()
