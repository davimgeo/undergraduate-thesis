import numpy as np 
import matplotlib.pyplot as plt

def plot_seismogram(
    seismogram: np.ndarray, 
    dt: float, 
    offset: int, 
    perc=99
) -> None:

  nt, nrec = seismogram.shape
  
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

def plot_model(model, perc=99) -> None:
  _, ax = plt.subplots(figsize=(12, 5))

  vmin = np.percentile(model, 100 - perc)
  vmax = np.percentile(model, perc)

  img = ax.imshow(
    model, aspect="auto", cmap="jet", 
    vmin=vmin, vmax=vmax
  )

  plt.colorbar(img, ax=ax, label="VP [m/s]")

  ax.set_xlabel("Distance [m]", fontsize=13)
  ax.set_ylabel("Depth [m]", fontsize=13)
  ax.set_title("Velocity Model", fontsize=13)

  plt.tight_layout()
  plt.show()

def plot_model_geometry(
  model: np.ndarray, 
  nb: int,
  dh: int, 
  recx: np.ndarray, 
  recz: np.ndarray, 
  srcx: np.ndarray, 
  srcz: np.ndarray,
  perc=99
) -> None:
  nzz, nxx = model.shape

  nz = nzz - 2*nb
  nx = nxx - 2*nb

  model = model[nb:nb + nz, nb:nb + nx]

  size  = (8, 10)
  if nxx > nzz: size = (10, 8)

  _, ax = plt.subplots(figsize=size)

  vmin = np.percentile(model, 100 - perc)
  vmax = np.percentile(model, perc)

  xloc = np.linspace(0, nx - 1, 11, dtype=int)
  xlab = np.array(xloc * dh, dtype=int)

  zloc = np.linspace(0, nz - 1, 7, dtype=int)
  zlab = np.array(zloc * dh, dtype=int)

  img = ax.imshow(
    model, aspect="auto", cmap="jet", 
    vmin=vmin, vmax=vmax
  )

  ax.plot(recx, recz / dh, 'gv', label="Receivers", markersize=12)
  ax.plot(srcx, srcz / dh, 'r*', markersize=12, label="Source")

  plt.colorbar(img, ax=ax, label="VP [m/s]")

  ax.set_yticks(zloc)
  ax.set_yticklabels(zlab)

  ax.set_xticks(xloc)
  ax.set_xticklabels(xlab)

  ax.set_xlabel("Distance [m]", fontsize=13)
  ax.set_ylabel("Depth [m]", fontsize=13)
  ax.set_title("Velocity Model", fontsize=13)

  plt.tight_layout()
  plt.show()

def plot1d(arr: np.ndarray) -> None:
  _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

  ax.plot(arr)
  
  plt.tight_layout()
  plt.show()

def plot1d_xy(x: np.ndarray, y: np.ndarray) -> None:
  _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

  ax.plot(x, y)
  
  plt.tight_layout()
  plt.show()

def plot1d_compare(arr: np.ndarray, arr2: np.ndarray) -> None:
  _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

  ax.plot(arr)
  ax.plot(arr2)
  
  plt.tight_layout()
  plt.show()

def plot2d(arr: np.ndarray) -> None:
  _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

  ax.imshow(arr, aspect="auto", cmap="jet")
  
  plt.tight_layout()
  plt.show()

def plot3d(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
  x_grid, y_grid = np.meshgrid(x, y)

  fig = plt.figure(figsize=(14, 6))

  ax1 = fig.add_subplot(1, 2, 1, projection="3d")
  ax2 = fig.add_subplot(1, 2, 2)

  surf = ax1.plot_surface(
    x_grid,
    y_grid,
    arr,
    cmap="viridis",
    alpha=0.8
  )

  ax1.contour(
    x_grid,
    y_grid,
    arr,
    zdir="z",
    offset=np.min(arr),
    levels=20,
    cmap="viridis"
  )

  ax1.set_xlabel(r"$x$")
  ax1.set_ylabel(r"$y$")
  ax1.set_zlabel(r"$f(x,y)$")
  ax1.view_init(elev=15, azim=4)

  im = ax2.imshow(
    arr,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis",
    aspect="auto"
  )

  ax2.set_xlabel(r"$x$")
  ax2.set_ylabel(r"$y$")

  fig.colorbar(surf, ax=ax1, shrink=0.6)
  fig.colorbar(im, ax=ax2)

  plt.tight_layout()
  plt.show()

def contourplot(
  Z,
  xmin,
  xmax,
  ymin,
  ymax,
  ncontours=50,
  fill=False
):
  row, col = Z.shape

  x = np.linspace(xmin, xmax, col)
  y = np.linspace(ymin, ymax, row)

  X, Y = np.meshgrid(x, y)

  plt.figure(figsize=(10, 8))

  if fill:
    plt.contourf(X, Y, Z, ncontours)
  else:
    plt.contour(X, Y, Z, ncontours)

  min_y_idx, min_x_idx = np.unravel_index(
    np.argmin(Z),
    Z.shape
  )

  min_x = x[min_x_idx]
  min_y = y[min_y_idx]

  plt.scatter(min_x, min_y, marker="x", s=50, color="r", label="Global Minimum")
  #plt.scatter(1, 1, marker="x", s=50, color="b", label="Analytical Global Minimum")

  plt.legend(loc="upper left")
  plt.tight_layout()
  plt.show()

def contourplot_opt(
  Z,
  x,
  y,
  xmin,
  xmax,
  ymin,
  ymax,
  ncontours=50,
  fill=False
):
  row, col = Z.shape

  x_axis = np.linspace(xmin, xmax, col)
  y_axis = np.linspace(ymin, ymax, row)

  X, Y = np.meshgrid(x_axis, y_axis)

  plt.figure(figsize=(10, 8))

  if fill:
    plt.contourf(X, Y, Z, ncontours)
  else:
    plt.contour(X, Y, Z, ncontours)

  min_y_idx, min_x_idx = np.unravel_index(
    np.argmin(Z),
    Z.shape
  )

  min_x = x_axis[min_x_idx]
  min_y = y_axis[min_y_idx]

  plt.scatter(x[0], y[0], marker="x", s=50, color="r",
              label=f"Initial Model: {x[0], y[0]}")
  plt.plot(
    x,
    y,
    marker="o",
    label="Optimization Path"
  )


  plt.plot(x[-1], y[-1], "-o", color="g", markersize=7,
           label=f"Final Model: {x[-1], y[-1]}")
  plt.scatter(min_x, min_y, marker="x", s=50, color="b",
              label=f"Global Minimum: {x_axis[min_x_idx], y_axis[min_y_idx]}")

  plt.legend(loc="upper left")
  plt.title(f"Objective Function | Number of iteration: {x.size - 2}")
  plt.tight_layout()
  plt.show()
