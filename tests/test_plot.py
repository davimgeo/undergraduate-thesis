import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"

def read_data(name):
    return np.fromfile(f"{DATA_DIR}/{name}.bin", dtype=np.float32)

def plot3d(Z, alphas, v0, title):
    Z = Z.reshape(len(v0), len(alphas))

    A, V = np.meshgrid(alphas, v0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        A,
        V,
        Z,
        cmap="viridis"
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$v_0$")
    ax.set_zlabel("Objective function")
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.show()

alphas = read_data("alphas_11")
v0 = read_data("v0_11")

l2 = read_data("l2_T02.5s_fmax_30")
l1 = read_data("l1_T02.5s_fmax_30")
cross = read_data("cross_T02.5s_fmax_30")
decon = read_data("decon_T02.5s_fmax_30")

plot3d(l2, v0, alphas, "L2")
plot3d(l1, v0, alphas, "L1")
plot3d(cross, v0, alphas, "Cross-correlation")
plot3d(decon, v0, alphas, "Deconvolution")

