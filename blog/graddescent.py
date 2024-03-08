import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def bowl(x, y):
    return x**2 + y**2


def dfbowl(x, y):
    return 2 * x, 2 * y


alpha = 0.1
steps = 25
x0, y0 = -8.0, 4.0

descent = [(x0, y0)]
x, y = x0, y0
for _ in range(steps):
    dx, dy = dfbowl(x, y)
    x, y = x - alpha * dx, y - alpha * dy
    descent.append((x, y))

xs, ys = zip(*descent)
zs = [bowl(x, y) for x, y in descent]

x_vals = np.linspace(-10, 10, 400)
y_vals = np.linspace(-10, 10, 400)
x_vals, y_vals = np.meshgrid(x_vals, y_vals)
z_vals = bowl(x_vals, y_vals)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")
ax.plot_surface(x_vals, y_vals, z_vals, cmap="cool", edgecolor="none", alpha=0.7)
ax.grid(False)

ax.plot(xs, ys, zs, color="red", marker="o", label="Loss")

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.zaxis.set_major_locator(plt.MaxNLocator(5))

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")

ax.set_xlabel("$x$", color="white")
ax.set_ylabel("$y$", color="white")

ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.tick_params(axis="z", colors="white")

plt.legend()
plt.savefig("./images/descent.png", dpi=300)
plt.show()
