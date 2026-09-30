import matplotlib.pyplot as plt
import numpy as np
  
def load(path: str, height: int, weight: int) -> np.ndarray:
  return np.fromfile(
    path, dtype=np.float32, count=height*weight
  ).reshape([height, weight], order='C')

nz = 141
nx = 681
dh = 2.5

nrec = 114
nsrc = 46

marmousi_real = load("data/FWI/marmousi_real_141x681x_dh25m.bin", nz, nx)
marmousi_inverted = load("data/FWI/m_7.bin", nz, nx)
marmousi_initial = load("data/FWI/m0.bin", nz, nx)

trace = 340

fig, ax = plt.subplots(
  nrows=1,
  ncols=2,
  figsize=(14, 6),
  gridspec_kw={"width_ratios": [1, 5]}
)

ax[0].plot(marmousi_real[:, trace], np.arange(nz), label="Real Model")
ax[0].plot(marmousi_inverted[:, trace], np.arange(nz), label="Inverted Model")
ax[0].plot(marmousi_initial[:, trace], np.arange(nz), label="Initial Model")

ax[0].set_xlabel("Velocity [m/s]")
ax[0].xaxis.set_label_position("top")
ax[0].xaxis.tick_top()

ax[0].set_ylabel("Depth [m]")

ax[0].grid(True)
ax[0].invert_yaxis()
ax[0].legend()

recx = np.linspace(0, nx - 1, nrec)
recz = 20 * np.ones(nrec)

srcx = np.linspace(0, nx - 1, nsrc)
srcz = 0 * np.ones(nsrc)

img = ax[1].imshow(marmousi_initial, aspect="auto", cmap="jet")
ax[1].axvline(trace, linestyle="--")

ax[1].plot(recx, recz, 'gv', label="Receivers", markersize=12)
ax[1].plot(srcx, srcz, 'r*', markersize=12, label="Source")

plt.colorbar(img, ax=ax[1], label="VP [m/s]")

ax[1].set_title("Initial Model", fontsize=13)

ax[1].legend()
plt.tight_layout()
plt.show()

