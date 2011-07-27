#!/usr/bin/python
#

from patch_util import *
from pylab import *
from omp import *
import logging 
import random

def ksvd(Y, K, T, D=None, max_err=0, max_iter=10):
  logger = logging.getLogger(__name__)
  print "btdubs, the ksvd logger is %s" % __name__

  (N, M) = Y.shape

  # if we're not given a dictionary for starters, make our own
  if D is None:
    D = rand(N, K)

  # normalize the dictionary regardless
  for i in range(K): 
    D[:,i] = D[:,i] / norm(D[:,i])

  logger.info("running ksvd on %d %d-dimensional vectors with K=%d"\
      % (M, N, K))

  # algorithm stuff
  X = zeros((K,N))
  err = inf
  iter_num = 0

  while iter_num < max_iter and err > max_err:
    # batch omp, woo!
    logger.info("staring batch omp...")
    X = omp_batch(D, Y, T, max_err)
    logger.info("omp complete!")
    logger.info(\
        'average l0 "norm" for ksvd iteration %d after omp was %f' \
        % (iter_num, len(nonzero(X)[0])/M) )

    # dictionary update -- protip: update dictionary columns in random
    # order
    atom_indices = range(K)
    random.shuffle(atom_indices)
    for (i, j) in zip(atom_indices, xrange(K)):
      if j % 25 == 0:
        logger.info("ksvd: iteration %d, updating atom %d of %d" \
            % (iter_num+1, j, K))

      # find nonzero entries
      x_using = nonzero(X[i,:])[0]

      if len(x_using) == 0:
        # unused column -> replace by signal in training data with worst
        # representation
        Repr_err = Y - dot(D,X)
        Repr_err_norms = [ norm(Repr_err[:,n]) for n in range(M) ]
        worst_signal = argmax(Repr_err_norms)
        D[:,i] = Y[:,worst_signal] / norm(Y[:,worst_signal])
        continue

      # compute residual error ... here's a trick passing almost all the
      # work to BLAS
      X[i,x_using] = 0
      Residual_err = Y[:,x_using] - dot(D,X[:,x_using])

      # update dictionary and weights -- sparsity-restricted rank-1
      # approximation
      U, s, Vt = svd(Residual_err)
      D[:,i] = U[:,0]
      X[i,x_using] = s[0]*Vt.T[:,0]

    # compute maximum representation error
    Repr_err = Y - dot(D, X)
    Repr_err_norms = [ norm(Repr_err[:,j]) for j in range(M) ]
    max_err = max(Repr_err_norms)

    # and increase the iter_num; repeat
    iter_num += 1

    # report a bit of info
    logger.info(\
        "maximum representation error for ksvd iteration %d was %f" \
        % (iter_num, max_err) )

  return D,X

