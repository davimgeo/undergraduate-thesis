import numpy as np
import matplotlib.pyplot as plt

def load(path: str, row: int, col: int, type=None):
    if type is None:
        type = np.float32

    return np.fromfile(
        path, dtype=type, count=row*col
    ).reshape(
        [row, col], order='C'
    )

ROW = 4001
COL = 21

arr = load("data/arr.bin", ROW, COL)

fft_arr = np.fft.fft2(arr)
fft_c_arr = load("data/fft.bin", ROW, COL, np.complex64)

ifft_arr = np.fft.ifft2(fft_arr)
ifft_c_arr = load("data/ifft.bin", ROW, COL)

fft_diff = np.abs(fft_arr - fft_c_arr)
ifft_diff = np.abs(ifft_arr - ifft_c_arr)

# =================================== PLOT =====================================

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

im0 = ax[0, 0].imshow(
    np.abs(fft_arr),
    aspect="auto",
    cmap="jet"
)
ax[0, 0].set_title("FFT - Python")
ax[0, 0].set_xlabel("Coluna")
ax[0, 0].set_ylabel("Linha")
fig.colorbar(im0, ax=ax[0, 0])

im1 = ax[0, 1].imshow(
    np.abs(fft_c_arr),
    aspect="auto",
    cmap="jet"
)
ax[0, 1].set_title("FFT - C")
ax[0, 1].set_xlabel("Coluna")
ax[0, 1].set_ylabel("Linha")
fig.colorbar(im1, ax=ax[0, 1])

im2 = ax[0, 2].imshow(
    fft_diff,
    aspect="auto",
    cmap="jet"
)
ax[0, 2].set_title(
    r"$|\mathrm{FFT}_{Python} - \mathrm{FFT}_{C}|$"
)
ax[0, 2].set_xlabel("Coluna")
ax[0, 2].set_ylabel("Linha")
fig.colorbar(im2, ax=ax[0, 2])

im3 = ax[1, 0].imshow(
    np.real(ifft_arr),
    aspect="auto",
    cmap="jet"
)
ax[1, 0].set_title("IFFT - Python")
ax[1, 0].set_xlabel("Coluna")
ax[1, 0].set_ylabel("Linha")
fig.colorbar(im3, ax=ax[1, 0])

im4 = ax[1, 1].imshow(
    ifft_c_arr,
    aspect="auto",
    cmap="jet"
)
ax[1, 1].set_title("IFFT - C")
ax[1, 1].set_xlabel("Coluna")
ax[1, 1].set_ylabel("Linha")
fig.colorbar(im4, ax=ax[1, 1])

im5 = ax[1, 2].imshow(
    ifft_diff,
    aspect="auto",
    cmap="jet"
)
ax[1, 2].set_title(
    r"$|\mathrm{IFFT}_{Python} - \mathrm{IFFT}_{C}|$"
)
ax[1, 2].set_xlabel("Coluna")
ax[1, 2].set_ylabel("Linha")
fig.colorbar(im5, ax=ax[1, 2])

plt.tight_layout()
plt.show()
