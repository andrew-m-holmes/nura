import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def main():
    n_million_range = np.arange(1e6, 10e6, 1e6)
    m_million_range = np.arange(1e6, 10e6, 1e6)

    n_million_grid, m_million_grid = np.meshgrid(n_million_range, m_million_range)
    complexity_million = 2 * n_million_grid * m_million_grid

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        n_million_grid, m_million_grid, complexity_million, cmap="cool"
    )

    ax.set_xlabel("$n$")
    ax.set_ylabel("$m$")
    ax.set_zlabel("$O(mn)$")

    ax.grid(False)

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.w_xaxis.line.set_color("white")
    ax.w_yaxis.line.set_color("white")
    ax.w_zaxis.line.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.tick_params(axis="z", colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")

    plt.savefig("./images/runtime.png", dpi=300)


if __name__ == "__main__":
    main()
