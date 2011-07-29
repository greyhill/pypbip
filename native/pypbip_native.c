#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "pypbip_omp.h"

/* module documentation */
PyDoc_STRVAR(
    pypbip_native__doc__,
    ""
    );

/* pypbip_omp_sf pydoc */
PyDoc_STRVAR(
    py_pypbip_omp_sf__doc__,
    "performs orthogonal matching pursuit (OMP) using a given dictionary\n"
    "\n"
    "parameters: (N, y, K, D, x, T, err); all ndarrays in Fortran order.\n"
    "\n"
    );
PyObject* py_pypbip_omp_sf(PyObject *self, PyObject *args) {
  int py_ok = true;
  bool alg_ok = true;

  PyArrayObject *y_ = NULL, 
                *D_ = NULL, 
                *x_ = NULL;
  float *y = NULL, 
        *D = NULL, 
        *x = NULL;
  int N, K, T;
  float err;

  /* parse arguments from python */
  py_ok = PyArg_ParseTuple(args, "iOiOOif", 
      &N, &y_, &K, &D_, &x_, &T, &err);
  if(!py_ok) {
    fprintf(stderr, "hm, problem.\n");
    PyErr_BadInternalCall();
  }

  y = PyArray_DATA(y_);
  D = PyArray_DATA(D_);
  x = PyArray_DATA(x_);

  Py_BEGIN_ALLOW_THREADS

  alg_ok = pypbip_omp_sf(
      N, y, K, D,
      x, T, err);

  Py_END_ALLOW_THREADS

  if(!alg_ok) {
    PyErr_BadInternalCall();
  }

  Py_INCREF(Py_None);
  return Py_None;
}

/* pypbip_omp_batch_sf */
PyDoc_STRVAR(
    py_pypbip_omp_batch_sf__doc__,
    "performs batched omp using a given dictionary\n\n");
PyObject* py_pypbip_omp_batch_sf(PyObject *self, PyObject *args) {
  int py_ok = true;
  bool alg_ok = true;

  PyArrayObject *y_ = NULL, 
                *D_ = NULL, 
                *x_ = NULL;
  float *y = NULL, 
        *D = NULL, 
        *x = NULL;
  int N, M, K, T;
  float err;

  /* parse arguments from python */
  py_ok = PyArg_ParseTuple(args, "iiOiOOif", 
      &N, &M, &y_, &K, &D_, &x_, &T, &err);
  if(!py_ok) {
    fprintf(stderr, "hm, problem.\n");
    PyErr_BadInternalCall();
  }

  y = PyArray_DATA(y_);
  D = PyArray_DATA(D_);
  x = PyArray_DATA(x_);

  Py_BEGIN_ALLOW_THREADS

  alg_ok = pypbip_omp_batch_sf(
      N, M, y, K, D,
      x, T, err);

  Py_END_ALLOW_THREADS

  if(!alg_ok) {
    PyErr_BadInternalCall();
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef pypbip_native_methods[] = {
  {"omp_sf", py_pypbip_omp_sf, METH_VARARGS,
    py_pypbip_omp_sf__doc__},
  {"omp_batch_sf", pypbip_omp_batch_sf, METH_VARARGS,
    py_pypbip_omp_batch_sf__doc__},
  {NULL, NULL}
};

PyMODINIT_FUNC
initpypbip_native() {
  Py_InitModule3("pypbip_native", pypbip_native_methods,
      pypbip_native__doc__);
}

