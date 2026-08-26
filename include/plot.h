#pragma once

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define PLOT_MODULE "plot"

#include "geometry.h"
#include "model.h"
#include "seismogram.h"

int plot_model(model_t* model);
int plot_seismogram(seismogram_t* seismogram, int offset);
int plot_model_geometry(model_t* model, int dh, geometry_t* geometry);
int plot1d(float* arr, int size);
int plot1d_compare(float* arr, float* arr2, int size);
int plot1d_xy(float* arr, float* arr2, int size);
int plot2d(float* arr, int row, int col);
int plot3d(float* arr, float* x, float* y, int row, int col);
int contourplot(
  float* arr,
  int row,
  int col,
  float xmin,
  float xmax,
  float ymin,
  float ymax
);
