import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def func(x1, x2):
    return x1**2 + x2**2


def grad_func(x1, x2):
    grad_x1 = 2 * x1
    grad_x2 = 2 * x2
    return np.array([grad_x1, grad_x2])


def gradient_descent(grad, start, learn_rate, n_iter=50, tol=1e-06):
    path = []
    x = start
    for _ in range(n_iter):
        grad_eval = grad(x[0], x[1])
        x_new = x - learn_rate * grad_eval
        if np.all(np.abs(x_new - x) <= tol):
            break
        x = x_new
        path.append(x)
    return np.array(path)


start = np.array([0.8, 0.8])
path = gradient_descent(grad_func, start, 0.1)

x1 = np.linspace(-1, 1, 400)
x2 = np.linspace(-1, 1, 400)
x1, x2 = np.meshgrid(x1, x2)
z = func(x1, x2)
levels = np.linspace(0, max(z.flatten()), 40)

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(x1, x2, z, levels=levels, cmap="cool")
ax.plot(path[:, 0], path[:, 1], "r.-", label="Gradient Descent Path")

ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.grid(False)

ax.set_xlabel("$x_1$", color="white")
ax.set_ylabel("$x_2$", color="white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

ax.legend(facecolor="black", edgecolor="white", framealpha=1)
plt.setp(plt.gca().get_legend().get_texts(), color="white")
plt.savefig("./images/descent.png", dpi=300)
