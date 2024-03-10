import nura
from nura.autograd.functional import jacfwd

def fn(a, b, c):
    return a * b + c


a = nura.rand(4)
b = nura.rand(4)
c = nura.tensor(1.0)
r = nura.rand(4)

output, jacobian = jacfwd((a, b, c), fn, pos=1)
print(f"output:\n{output}\n\njacobian:\n{jacobian}")





