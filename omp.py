from patch_util import *
from pylab import *
import logging 

def omp_stopfun(max_iter, min_err):
  return lambda n, err: n > max_iter or err < min_err

def orthogonal_matching_pursuit(D, y, stop_fun, inner_product=None):
  """Greedy algorithm for finding a sparse representation of y on
  overcomplete and potentially redundant dictionary D.

  inputs:
  D -- R^{M x K} overcomplete dictionary
  y -- R^{M} vector
  stop_fun -- (iter_num, rel_residual_norm) -> bool function; true to
    stop
  inner_product -- (v,v) -> R^+ u {0}, an inner product; default is
    standard euclidean inner product

  output:
  x -- R^{K} vector such that ||Dx - y||_2 is small

  """
  from scipy.linalg.decomp_lu import lu_solve

  (M, K) = D.shape
  x = None
  iter_num = 1
  err = inf
  residual = y.copy()
  used_atoms = []

  # qr factorization of orthogonal projection onto selected atoms
  Q = None
  R = None

  def cond_argmax(invalid, collection, x, y):
    if y in invalid:
      return x
    elif x in invalid:
      return y
    elif collection[y] > collection[x]:
      return y
    else:
      return x

  while not stop_fun(iter_num, err / norm(y) ):
    # find max inner product
    iprods = None
    if inner_product is None:
      iprods = dot(residual.reshape((1, M)), D).squeeze()
    else:
      iprods = array(map(lambda v: inner_product(residual, v), \
        [ D[:,i] for i in range(K) ] ))
    # basically, run argmax over a subset without explicitly creating
    # the subset
    max_atom = reduce( \
        lambda x,y: cond_argmax(used_atoms, iprods, x, y), \
        range(len(iprods)) )
    used_atoms.append(max_atom)

    new_atom = D[:,max_atom].copy()

    from math import isnan

    if Q is None:
      # first iteration
      Q = new_atom.reshape( (M, 1) )
      R = ones((1,1))
    else:
      # augment the QR factorization
      R_new = zeros( (iter_num, iter_num) )
      R_new[0:iter_num-1, 0:iter_num-1] = R
      R = R_new
      for j in range(iter_num-1):
        # orthogonalize 
        r = 0
        if inner_product is None:
          r = inner(new_atom, Q[:,j])
        else:
          r = inner_product(new_atom, Q[:,j])
        R[j, iter_num-1] = r
        new_atom -= r * Q[:,j]
      R[iter_num-1, iter_num-1] = norm(new_atom)

      # work around the case where we have perfect representation
      new_atom_norm = 0
      if inner_product is None:
        new_atom_norm = norm(new_atom)
      else:
        new_atom_norm = sqrt( inner_product(new_atom, new_atom) )
      if allclose(new_atom_norm, 0):
        # roll back this iteration a bit TODO make this cleaner
        used_atoms = used_atoms[:-1]
        R = R[0:iter_num-1, 0:iter_num-1]
        break

      Q = hstack((Q, new_atom.reshape( (M,1) )))
      z = new_atom
   
    # update residual, coeffs and iteration count
    r = inner(new_atom.squeeze(), residual.squeeze())

    if x is None:
      x = array([r])
    else:
      x = vstack((x, r))
    residual -= (r*new_atom).reshape(residual.shape)
    iter_num += 1

    # compute R^{-1} x, the dictionary coefficients
    if iter_num > 2:
      coeffs = None
      coeffs = lu_solve( (R, arange(R.shape[0]) ), x )
      
      q = zeros( (K,1) )
      q[(used_atoms)] = coeffs

      err = norm(y - dot(D,q))

  # compute R^{-1} x, the dictionary coefficients
  assert(iter_num > 1)
  coeffs = lu_solve( (R, arange(R.shape[0]) ), x )
  q = zeros( (K,1) )
  q[(used_atoms)] = coeffs
  return q

