from patch_util import *
from pylab import *
import logging 

def omp(D, y, T, err):
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
  err = float32(err)
  x = zeros((K,1), dtype='float32', order='F')

  omp_sf(N, yf, K, Df, x, T, err)

  return x

def omp_batch(D, Y, T, err):
  """Perform omp on a collection of vectors arranged as columns in a
  matrix.

  Performs OMP on a collection of vectors arranged as columns in a
  matrix.  Potentially faster than running OMP on each vector
  individually.

  """
  (N, M) = Y.shape
  K = D.shape[1]
  X = zeros((K, M))
  for i in xrange(M):
    X[:,i] = omp(D, Y[:,i], T, err).flatten()
  return X

