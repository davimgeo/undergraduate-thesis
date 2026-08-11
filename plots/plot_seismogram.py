import sys

import numpy as np
import matplotlib.pyplot as plt

def load(PATH: str, height: int, width: int) -> np.ndarray:
  return np.fromfile(
    PATH, dtype=np.float32, count=height*width
  ).reshape([height, width], order='C')

def plot_seismogram(
    seismogram: np.ndarray, 
    nt: int, dt: float, 
    offset: int, 
    nrec: int, 
    perc=99
) -> None:
  
  tloc = np.linspace(0, nt - 1, 11, dtype=int)
  tlab = np.around(tloc * dt, decimals=1)

  xloc = np.linspace(0, nrec - 1, 9)
  xlab = np.array(offset * 10 * xloc, dtype=int)

  scale_min = np.percentile(seismogram, 100 - perc)
  scale_max = np.percentile(seismogram, perc)

  fig, ax = plt.subplots(figsize=(10, 8))

  img = ax.imshow(seismogram, aspect="auto", cmap="Greys",
                    vmin=scale_min, vmax=scale_max)

  ax.set_yticks(tloc)
  ax.set_yticklabels(tlab)

  ax.set_xticks(xloc)
  ax.set_xticklabels(xlab)

  ax.set_xlabel("Offset (m)", fontsize=13)
  ax.set_ylabel("TWT (s)", fontsize=13)

  plt.show()

nt = 2001
nrec = 41

if len(sys.argv) > 1:
  PATH = str(sys.argv[1])
  nt   = int(sys.argv[2])
  nrec = int(sys.argv[3])

seismogram = load("data/seismogram.bin", nt, nrec)

plot_seismogram(seismogram, nt, 1e-3, 5, nrec)
