import numpy as np 
import matplotlib.pyplot as plt 

SIZE = 51

REF_ALPHA = 3.5

alphas = np.fromfile("data/alphas_51.bin", dtype=np.float32, count=SIZE)
da     = alphas - REF_ALPHA

cross = np.fromfile("data/cross_51.bin", dtype=np.float32, count=SIZE)
decon = np.fromfile("data/decon_51.bin", dtype=np.float32, count=SIZE)
l2    = np.fromfile("data/l2_51.bin", dtype=np.float32, count=SIZE)
l1    = np.fromfile("data/l1_51.bin", dtype=np.float32, count=SIZE)

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(da, cross, label="Cross-Correlation")
ax.plot(da, decon, label="Deconvolution")
ax.plot(da, l2, label="L2")
ax.plot(da, l1, label="L1")

ax.set_xlabel(r"$\Delta alphas$", fontsize=13)
ax.set_ylabel("Normalized Objective Function", fontsize=13)

ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
