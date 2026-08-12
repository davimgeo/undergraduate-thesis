#pragma once

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define PLOT_MODULE "plot"

static int plot_python_initialized = 0;

static inline int plot_python_init(void)
{
  if (plot_python_initialized) return 0;

  Py_Initialize();
  if (!Py_IsInitialized()) return -1;
  
  if (_import_array() < 0) 
  {
    PyErr_Print();
    Py_Finalize();
    return -1;
  }

  PyRun_SimpleString(
      "import sys, os; "
      "sys.path.insert(0, os.path.expanduser('~/plots/src'))"
  );

  plot_python_initialized = 1;

  return 0;
}

static inline void plot_python_finalize(void)
{
  if (!plot_python_initialized) return;

  Py_Finalize();

  plot_python_initialized = 0;
}

static inline int plot_model(float *model, int nx, int nz)
{
  if (plot_python_init() != 0) return -1;

  PyObject *name = PyUnicode_FromString(PLOT_MODULE);

  if (!name) 
  {
    PyErr_Print();
    return -1;
  }

  PyObject *module = PyImport_Import(name);

  Py_DECREF(name);

  if (!module) 
  {
    PyErr_Print();
    return -1;
  }

  PyObject *func = PyObject_GetAttrString(module, "plot_model");
  if (!func || !PyCallable_Check(func)) 
  {
    PyErr_Print();
    Py_XDECREF(func);
    Py_DECREF(module);
    return -1;
  }

  npy_intp dims[2] = {nz, nx};

  PyObject *py_model =
      PyArray_SimpleNewFromData(
          2,
          dims,
          NPY_FLOAT32,
          model
      );

  if (!py_model) 
  {
    PyErr_Print();
    Py_DECREF(func);
    Py_DECREF(module);
    return -1;
  }

  PyObject *args = PyTuple_Pack(1, py_model);

  if (!args) 
  {
    PyErr_Print();
    Py_DECREF(py_model);
    Py_DECREF(func);
    Py_DECREF(module);
    return -1;
  }

  PyObject *result = PyObject_CallObject(func, args);
  if (!result) PyErr_Print();

  Py_XDECREF(result);
  Py_DECREF(args);
  Py_DECREF(py_model);
  Py_DECREF(func);
  Py_DECREF(module);

  return result ? 0 : -1;
}

static inline int plot_seismogram(
    float *seismogram,
    int nt,
    int nrec,
    float dt,
    int offset
)
{
    if (plot_python_init() != 0) return -1;

    PyObject *name = PyUnicode_FromString(PLOT_MODULE);
    if (!name) 
    {
      PyErr_Print();
      return -1;
    }

    PyObject *module = PyImport_Import(name);

    Py_DECREF(name);

    if (!module) 
    {
      PyErr_Print();
      return -1;
    }

    PyObject *func =
        PyObject_GetAttrString(module, "plot_seismogram");

    if (!func || !PyCallable_Check(func)) 
    {
      PyErr_Print();
      Py_XDECREF(func);
      Py_DECREF(module);
      return -1;
    }

    npy_intp dims[2] = {nt, nrec};

    PyObject *py_seismogram =
        PyArray_SimpleNewFromData(
            2,
            dims,
            NPY_FLOAT32,
            seismogram
        );

    if (!py_seismogram) 
    {
      PyErr_Print();
      Py_DECREF(func);
      Py_DECREF(module);
      return -1;
    }

    PyObject *dt_obj = PyFloat_FromDouble(dt);

    PyObject *offset_obj = PyLong_FromLong(offset);

    if (!dt_obj || !offset_obj) 
    {
      PyErr_Print();

      Py_XDECREF(dt_obj);
      Py_XDECREF(offset_obj);

      Py_DECREF(py_seismogram);
      Py_DECREF(func);
      Py_DECREF(module);

      return -1;
    }

    PyObject *args =
        PyTuple_Pack(
            3,
            py_seismogram,
            dt_obj,
            offset_obj
        );

    Py_DECREF(dt_obj);
    Py_DECREF(offset_obj);

    if (!args) 
    {
      PyErr_Print();

      Py_DECREF(py_seismogram);
      Py_DECREF(func);
      Py_DECREF(module);

      return -1;
    }

    PyObject *result = PyObject_CallObject(func, args);
    if (!result) PyErr_Print();

    Py_XDECREF(result);
    Py_DECREF(args);
    Py_DECREF(py_seismogram);
    Py_DECREF(func);
    Py_DECREF(module);

    return result ? 0 : -1;
}

static inline int err_name(PyObject* name)
{
  if (!name)
  {
    PyErr_Print();
    return -1;
  }
  return 0;
}

static inline int plot_model_geometry(
  const float *model,
  int nz,
  int nx,
  int nb,
  int dh,
  const float *recx,
  const float *recz,
  int nrec,
  const float *srcx,
  const float *srcz,
  int nsrc
)
{
  if (plot_python_init() != 0) return -1;

  PyObject *name = PyUnicode_FromString(PLOT_MODULE);
  err_name(name);

  PyObject *module = PyImport_Import(name);
  Py_DECREF(name);

  if (!module)
  {
    PyErr_Print();
    return -1;
  }

  PyObject *func = PyObject_GetAttrString(
    module,
    "plot_model_geometry"
  );

  if (!func || !PyCallable_Check(func))
  {
    PyErr_Print();
    Py_XDECREF(func);
    Py_DECREF(module);
    return -1;
  }

  npy_intp model_dims[2] = {nz, nx};

  PyObject *py_model = PyArray_SimpleNewFromData(
    2,
    model_dims,
    NPY_FLOAT32,
    (void *)model
  );

  npy_intp rec_dims[1] = {nrec};

  PyObject *py_recx = PyArray_SimpleNewFromData(
    1,
    rec_dims,
    NPY_FLOAT32,
    (void *)recx
  );

  PyObject *py_recz = PyArray_SimpleNewFromData(
    1,
    rec_dims,
    NPY_FLOAT32,
    (void *)recz
  );

  npy_intp src_dims[1] = {nsrc};

  PyObject *py_srcx = PyArray_SimpleNewFromData(
    1,
    src_dims,
    NPY_FLOAT32,
    (void *)srcx
  );

  PyObject *py_srcz = PyArray_SimpleNewFromData(
    1,
    src_dims,
    NPY_FLOAT32,
    (void *)srcz
  );

  if (!py_model || !py_recx || !py_recz || !py_srcx || !py_srcz)
  {
    PyErr_Print();

    Py_XDECREF(py_model);
    Py_XDECREF(py_recx);
    Py_XDECREF(py_recz);
    Py_XDECREF(py_srcx);
    Py_XDECREF(py_srcz);

    Py_DECREF(func);
    Py_DECREF(module);

    return -1;
  }

  PyObject *py_nz = PyLong_FromLong(nz);
  PyObject *py_nx = PyLong_FromLong(nx);
  PyObject *py_dh = PyLong_FromLong(dh);
  PyObject *py_nb = PyLong_FromLong(nb);

  if (!py_nz || !py_nx || !py_dh)
  {
    PyErr_Print();

    Py_XDECREF(py_nz);
    Py_XDECREF(py_nx);
    Py_XDECREF(py_dh);
    Py_XDECREF(py_nb);

    Py_DECREF(py_model);
    Py_DECREF(py_recx);
    Py_DECREF(py_recz);
    Py_DECREF(py_srcx);
    Py_DECREF(py_srcz);

    Py_DECREF(func);
    Py_DECREF(module);

    return -1;
  }

  PyObject *args = PyTuple_Pack(
    7,
    py_model,
    py_nb,
    py_dh,
    py_recx,
    py_recz,
    py_srcx,
    py_srcz
  );

  Py_DECREF(py_nz);
  Py_DECREF(py_nx);
  Py_DECREF(py_dh);

  if (!args)
  {
    PyErr_Print();

    Py_DECREF(py_model);
    Py_DECREF(py_recx);
    Py_DECREF(py_recz);
    Py_DECREF(py_srcx);
    Py_DECREF(py_srcz);

    Py_DECREF(func);
    Py_DECREF(module);

    return -1;
  }

  PyObject *result = PyObject_CallObject(func, args);

  if (!result)
  {
    PyErr_Print();
  }

  Py_XDECREF(result);
  Py_DECREF(args);

  Py_DECREF(py_model);
  Py_DECREF(py_recx);
  Py_DECREF(py_recz);
  Py_DECREF(py_srcx);
  Py_DECREF(py_srcz);

  Py_DECREF(func);
  Py_DECREF(module);

  return result ? 0 : -1;
}




