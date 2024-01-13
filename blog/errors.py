import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.power(x, 2)  

def df(x):
    return 2 * x

def f(x):
    return 64 * x * (1 - x) * (1 - 2*x)**2 * (1 - 8*x + 8*x**2)**2

def df(x):
    return (64*x*(1 - 2*x)**2*(1 - x)*(32*x - 16)*(8*x**2 - 8*x + 1) - 
            64*x*(1 - 2*x)**2*(8*x**2 - 8*x + 1)**2 + 
            64*x*(1 - x)*(8*x - 4)*(8*x**2 - 8*x + 1)**2 + 
            64*(1 - 2*x)**2*(1 - x)*(8*x**2 - 8*x + 1)**2)

x0 = 2.
hs = np.logspace(-7, 0, 100).astype(np.float32)  

true_grad = df(x0)
forward_grad = (f(x0 + hs) - f(x0)) / hs
central_grad = (f(x0 + hs) - f(x0 - hs)) / (2 * hs)
central_difference_error = abs(true_grad - central_grad)
forward_difference_error = abs(true_grad - forward_grad)

plt.style.use("dark_background")
plt.loglog(hs, central_difference_error, label="Central Difference")
plt.loglog(hs, forward_difference_error, label="Forward Difference\nDerivative")
plt.xlabel("h")
plt.ylabel("Error")
plt.axhline(0, color="white")  
plt.axvline(0, color="white")  
plt.legend()
plt.show()
