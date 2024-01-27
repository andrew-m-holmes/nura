import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return (x - 10) ** 2 * (3 * x**2 - 3 * x + 1)


def df(x):
    return 2 * (x - 10) * (3 * x**2 - 3 * x + 1) + (x - 10) ** 2 * (6 * x - 3)


x0 = 2.0
hs = np.logspace(-7, 0, 30).astype(np.float32)

true_grad = df(x0)
forward_grad = (f(x0 + hs) - f(x0)) / hs
central_grad = (f(x0 + hs) - f(x0 - hs)) / (2 * hs)
central_difference_error = abs(true_grad - central_grad)
forward_difference_error = abs(true_grad - forward_grad)

plt.style.use("dark_background")
plt.loglog(
    hs, central_difference_error, "o", label="Central Difference", color="dodgerblue"
)
plt.loglog(hs, forward_difference_error, "o", label="Forward Difference", color="red")
plt.text(1.75e-6, 0.1, "Round-off")
plt.text(5e-3, 0.1, "Truncation")
plt.xlabel("h")
plt.ylabel("Error")
plt.axhline(0, color="white")
plt.axvline(0, color="white")
plt.legend()
plt.savefig("errors.png", dpi=300)
