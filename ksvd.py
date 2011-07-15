#!/usr/bin/python
#
# ksvd, omp python implementations (c) Madison McGaffin, 2011

from patch_util import *
from pylab import *
from omp import *
import logging 

def ksvd(Y, k, pursuit_fun, stop_fun):
  """Compute a dictionary capable of sparsely representing the given
  data.

  Let Y be R^{N x M}.  ksvd computes a R^{N x k} dictionary and a R^{k x
  M} set of coefficients such that X satisfies the sparsity constraints
  imposed by the given pursuit function.  ksvd attempts to minimize the 
  representation error ||Y - DX||_F.

  inputs:
  Y -- training data
  k -- number of atoms in the computed dictionary
  pursuit_fun(D, y) -- return a sparse R^{k x 1} vector, x, with small
    representation error ||y - Dx||_2.  See also: matching_pursuit.
  stop_fun -- (iter_num, err) -> whether or not to stop at this point
  
  outputs:
  D, X
  """
  (N, M) = Y.shape
  logger = logging.getLogger("ksvd")
  logger.info("running ksvd on %d %d-dimensional training vectors with k=%d"\
      % (M, N, k))


  # initialize dictionary columns; normalize
  D = rand(N, k)
  D[:,0] = 1
  for j in range(k): D[:,j] = D[:,j] / norm(D[:,j])

  # initialize coefficients
  # consider using sparse representation of X?
  X = zeros((k, M))

  iter_num = 1

  import ghpmath
  ticker = ghpmath.ticker(2)
  err = inf
  total_energy = 1

  while not stop_fun(iter_num, err/total_energy):
    logger.info("beginning iteration %d" % iter_num)
    err = 0
    total_energy = 0

    # use pursuit to find sparse columns of X
    for i in range(M): 
      if ticker.due(): logger.info("matching pursuit on column %d" % i)
      x = pursuit_fun(D, Y[:,i])
      X[:,i] = x.squeeze()

    # dictionary update
    for i in range(k):
      if i==0: continue
      if ticker.due(): logger.info("dictionary update on column %d" % i)

      x_using = nonzero(X[i,:])[0]
      if len(x_using) == 0: continue

      # compute residual error
      Residual_err = Y[:,x_using].copy()
      for j in range(k):
        if i==j: continue
        a = D[:,j]
        b = X[j,x_using].squeeze()
        Residual_err -= outer(a,b)

      # find optimal rank-1 approximation to residual error
      try:
        U, s, V = svd(Residual_err)
        err += sum( s[1:]**2 ) 
        total_energy += sum(s**2)
        # update dictionary and weights
        D[:,i] = U[:,0]
        X[i,x_using] = s[0]*V[:,0]
      except LinAlgError:
        logger.warning("svd failure updating column %d on iter %d" \
            % (i, iter_num))
      
    logger.info('percent representation error for iteration %d was %f' % (iter_num,\
      err/total_energy))
    logger.info('average zero-norm: %f' % (len(nonzero(X)[0]) / k))
    iter_num += 1

  return D,X

