import jax
import jax.numpy as jnp
import numpy as np
import deepnet as dn
from deepnet.autograd.functional import vjp, jvp, grad

def main():

    a = np.random.rand(4)
    b = np.random.rand(4)
    c = np.random.rand(1)
    v = np.ones(4)
    u = np.ones(1)
    
    dn_a = dn.tensor(a, usegrad=True)
    dn_b = dn.tensor(b, usegrad=True)
    dn_c = dn.tensor(c, usegrad=True)
    dn_v = dn.tensor(v)
    dn_u = dn.tensor(u)

    def f(a, b, c):
        return a * b + c 

    primals, tangents = jax.jvp(f, (a, b, c), (v, v, u))
    print(primals)
    print(tangents)

    primals, tangents = jvp((dn_a, dn_b, dn_c), (dn_v, dn_v, dn_u), f)
    print(primals)
    print(tangents)

    primals, jax_vjp_f = jax.vjp(f, a, b, c)
    cotangents = jax_vjp_f(v)
    print(primals)
    print(cotangents)

    primals, cotangents = vjp((dn_a, dn_b, dn_c), dn_v, f)
    print(primals)
    print(cotangents)

if __name__ == "__main__":
    main()
