import neuro
from neuro.autograd.functional import jacfwd

def fn(a, b, c):
    return a * b + c


a = neuro.rand(4)
b = neuro.rand(4)
c = neuro.tensor(1.0)
r = neuro.rand(4)

output, jacobian = jacfwd((a, b, c), fn, pos=1)
print(f"output:\n{output}\n\njacobian:\n{jacobian}")





