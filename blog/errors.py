import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.power(x, 2)  

def df(x):
    return 2 * x

x0 = 2.
hs = np.logspace(-7, -1, 150).astype(np.float32)  

true_grad = df(x0)
forward_grad = (f(x0 + hs) - f(x0)) / hs
# central_grad = (f(x0 + hs) - f(x0 - hs)) / (2 * hs)
# central_difference_error = abs(true_grad - central_grad)
forward_difference_error = abs(true_grad - forward_grad)

plt.style.use("dark_background")
# plt.loglog(hs, central_difference_error, label="Central Difference")
plt.loglog(hs, forward_difference_error, label="Forward Difference\nDerivative")
plt.xlabel("h")
plt.ylabel("Error")
plt.axhline(0, color="white")  
plt.axvline(0, color="white")  
plt.title("Finite Difference Approximation Error")
plt.legend()
plt.show()
