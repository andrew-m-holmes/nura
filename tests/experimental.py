import deepnet as dn
from deepnet.autograd.functional import grad, vjp, jvp


def main():
    
    a = dn.tensor(4, usegrad=True, dtype=dn.float)
    b = dn.tensor(3, usegrad=True, dtype=dn.float)
    c = dn.tensor(1, usegrad=True, dtype=dn.float)

    def f(a, b, c):
        return a * b + c

    out = f(a, b, c)
    
    grads = grad((a, b, c), out)
    print(grads)

    out.backward()
    print(a.grad, b.grad, c.grad)

    vec = dn.tensor(1, usegrad=True, dtype=dn.float)
    grads = vjp((a, b, c), vec, f)
    print(grads)

    vec = (vec, vec, vec)
    grads = jvp((a, b, c), vec, f)
    print(grads)

if __name__ == "__main__":
    main()
