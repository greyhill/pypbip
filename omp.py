from patch_util import *
from pylab import *
import logging 

def omp(D, Y, T, max_err):
  """Greedy algorithm for finding a sparse representation of y on the
  overcomplete dictionary D.
  
  Attempts to solve the problem argmin_x ||Dx - y||_2 such that 
  ||x||_0 <= T.  Less than T coefficients will be used if y can be
  represented with residual error less than err.

  Note: Y can be a vector or a matrix.  This routine calls either
  omp_single or omp_many depending on the shape of Y.

  """
  if len(Y.shape) < 2 or Y.shape[1] == 1:
    return omp_single(D, Y, T, max_err)
  else:
    return omp_many(D, Y, T, max_err)

def omp_single(D, y, T, max_err):
  """Greedy algorithm for finding a sparse representation of y on the
  overcomplete dictionary D.
  
  Attempts to solve the problem argmin_x ||Dx - y||_2 such that 
  ||x||_0 <= T.  Less than T coefficients will be used if y can be
  represented with residual error less than err.

  """
  from numpy import array, int32, float32
  from native import omp_sf

  N = y.shape[0]
  K = D.shape[1]

  N = int32(N)
  K = int32(K)
  Df = array(D, dtype='float32', order='F')
  yf = array(y, dtype='float32', order='F')
  T = int32(T)
  max_err = float32(max_err)
  x = zeros((K,1), dtype='float32', order='F')

  omp_sf(N, yf, K, Df, x, T, max_err)

  return x

def omp_many(D, Y, T, max_err):
  """Perform omp on a collection of vectors arranged as columns in a
  matrix.

  Performs OMP on a collection of vectors arranged as columns in a
  matrix.  Potentially faster than running OMP on each vector
  individually.

  Note: this is NOT the "batch OMP" algorithm, but a 

  """
  from numpy import array, int32, float32
  from native import omp_batch_sf

  N = Y.shape[0]
  M = Y.shape[1]
  K = D.shape[1]
  assert(N == D.shape[0])

  N = int32(N)
  M = int32(M)
  K = int32(K)
  Df = array(D, dtype='float32', order='F')
  Yf = array(Y, dtype='float32', order='F')
  T = int32(T)
  max_err = float32(max_err)
  X = zeros((K,M), dtype='float32', order='F')

  omp_batch_sf(N, M, Yf, K, Df, X, T, max_err)

  return X

