import nura
from nura.autograd.functional import jacfwd

def fn(a, b, c):
    return a * b + c

a = nura.tensor([1.0, 2.0, 3.0, 4.0])
b = nura.tensor([5.0, 6.0, 7.0, 8.0])
c = nura.tensor(1.0)
r = nura.ones(4).double()

output, jacobian = jacfwd((a, b, c), fn, pos=1)
print(f"output:\n{output}\n\njacobian:\n{jacobian}")
