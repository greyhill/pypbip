import ctypes as ct
import os.path
import numpy as np

__mydir = os.path.dirname(os.path.realpath(__file__))
__lib = ct.cdll.LoadLibrary(__mydir + os.path.sep + "pypbip_native.so")

__omp_sf = __lib.pypbip_omp_sf
__omp_sf.restype = ct.c_bool
__omp_sf.argtypes = [ ct.c_int32, 
    np.ctypeslib.ndpointer(dtype = ct.c_float),
    ct.c_int32,
    np.ctypeslib.ndpointer(dtype = ct.c_float),
    np.ctypeslib.ndpointer(dtype = ct.c_float),
    ct.c_int,
    ct.c_float ]

__omp_batch_sf = __lib.pypbip_omp_batch_sf
__omp_batch_sf.restype = ct.c_bool
__omp_batch_sf.argtypes = [ \
    ct.c_int,
    ct.c_int,
    np.ctypeslib.ndpointer(dtype = ct.c_float),
    ct.c_int,
    np.ctypeslib.ndpointer(dtype = ct.c_float),
    np.ctypeslib.ndpointer(dtype = ct.c_float),
    ct.c_int,
    ct.c_float ]

def omp(D, Y, T, max_err):
  """Greedy algorithm for finding a sparse representation of y on the
  overcomplete dictionary D.
  
  Attempts to solve the problem argmin_x ||Dx - y||_2 such that 
  ||x||_0 <= T.  Less than T coefficients will be used if y can be
  represented with residual error less than err.

  Note: Y can be a vector or a matrix.  This routine calls either
  omp_single or omp_many depending on the shape of Y.

  """
  import numpy.ctypeslib as npct

  N = ct.c_int32(D.shape[0])
  K = ct.c_int32(D.shape[1])

  Y = np.array(Y, dtype=np.float32, order='F')

  D = np.array(D, dtype=np.float32, order='F')

  T = ct.c_int32(T)
  max_err = ct.c_float(max_err)

  if len(Y.shape) < 2 or Y.shape[1] == 1:
    # Y is a vector
    X = np.empty(shape=(D.shape[1],1), dtype=np.float32, order='F')
    if not __omp_sf(N, Y, K, D, X, T, max_err):
      # error
      raise Exception("something happened whilst executing pypbip_omp_sf")
    else:
      # ok
      return X
  else:
    # Y is a matrix; use batch
    M = ct.c_int32(Y.shape[1])
    X = np.empty(shape=(D.shape[1], Y.shape[1]), dtype=np.float32, order='F')
    if not __omp_batch_sf(N, M, Y, K, D, X, T, max_err):
      raise Exception(\
          "something happened whilst executing pypbip_omp_batch_sf")
    else:
      return X

