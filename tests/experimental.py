import deepnet as dn
from deepnet.autograd.functional import grad, vjp, jvp


def main():
    
    a = dn.rand(4, usegrad=True, dtype=dn.float)
    b = dn.rand(4, usegrad=True, dtype=dn.float)
    c = dn.rand(1, usegrad=True, dtype=dn.float)

    def f(a, b, c):
        return a * b + c
    
    cot = dn.ones(4, dtype=dn.float)
    cots = vjp((a, b, c), cot, f)
    print(cots)

    out = f(a, b, c)
    grads = grad((a, b, c), out, cot * 10)
    print(grads)

    out.backward(cot * 100) 
    print(a.grad, b.grad, c.grad)
    print(out)
    
    tans = (dn.oneslike(a), dn.oneslike(b), dn.oneslike(c))
    tan = jvp((a, b, c), tans, f)    
    print(tan)

if __name__ == "__main__":
    main()
