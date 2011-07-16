#!/usr/bin/python
#
# ksvd, omp python implementations (c) Madison McGaffin, 2011

from patch_util import *
from pylab import *
from omp import *
import logging 
import random

def ksvd_stopfun(max_iter, min_err):
  return lambda n, err: n > max_iter or err < min_err

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
  D = rand(N,k)
  for j in range(k): D[:,j] = D[:,j] / norm(D[:,j])

  # initialize coefficients
  # consider using sparse representation of X?
  X = zeros((k, M))

  iter_num = 1

  err = inf
  total_energy = norm(Y, 'fro')

  while not stop_fun(iter_num, err/total_energy):
    logger.info("beginning iteration %d" % iter_num)
    err = 0

    # use pursuit to find sparse columns of X
    for i in range(M): 
      x = pursuit_fun(D, Y[:,i])
      X[:,i] = x.squeeze()

    err = norm(Y - dot(D,X), 'fro')
    logger.info(('percent representation error for iteration %d was %f ' \
        + 'before dictionary update') % (iter_num, err/total_energy))

    # dictionary update; perform in random order
    atom_indices = range(k)
    random.shuffle(atom_indices)
    for i in atom_indices:
      x_using = nonzero(X[i,:])[0]

      if len(x_using) == 0: 
        # unused column -> replace by the signal in the training data
        # with the worst representation
        Repr_err = Y - dot(D,X)
        Repr_err_norms = [ norm(Repr_err[:,j]) for j in range(M) ]
        worst_signal = argmax(Repr_err_norms)
        D[:,i] = Y[:,worst_signal] / norm(Y[:,worst_signal])
        continue

      # compute residual error
      # threw out the following two techniques for numerical
      # stability/speed reasons
      #Residual_err = Y[:,x_using].copy()
      #for j in range(k):
      #  if i==j: continue
      #  a = D[:,j]
      #  b = X[j,x_using]
      #  Residual_err -= outer(a,b)
      #Residual_err = Y - dot(D, X) + outer(D[:,i], X[i,:])
      #Residual_err = Residual_err[:,x_using]

      X[i,:] = 0
      Residual_err = Y[:,x_using] - dot(D,X[:,x_using])

      # find optimal rank-1 approximation to residual error
      try:
        U, s, V = svd(Residual_err)
        # update dictionary and weights
        D[:,i] = U[:,0]
        X[i,x_using] = s[0]*V[0,:]
        Residual_err = Y[:,x_using] - dot(D,X[:,x_using])
      except LinAlgError:
        logger.warning("svd failure updating column %d on iter %d" \
            % (i, iter_num))

    # compute representation error
    err = norm(Y - dot(D,X), 'fro')

    logger.info('percent representation error for iteration %d was %f' % (iter_num,\
      err/total_energy))
    logger.info('average zero-norm: %f' % (len(nonzero(X)[0]) / M))
    iter_num += 1

  return D,X

