def ksvd_denoise(alpha, u, pg, T, K, max_err=0, max_iter=10,
    D = None,
    ksvd_approx=False, preserve_dc=False):
  """Attempts to denoise the image u using ksvd and some sort of
  "sparsity prior".

  This implements something close to the denoising algorithm described
  in Michael Elad and Michal Aharon's 2006 paper "Image Denoising via
  Sparse and Redundant Representations over Learned Dictionaries" using
  the noisy image as a prior.

  As per the paper, it's _very important_ to set T, the maximum sparse
  coefficient l0-norm and max_error according to the estimated noise level
  of the image.  In the paper mentioned above, setting max_err = 1.15 * sigma
  is recommended, and alpha = 30/sigma ... assuming AWGN.

  """
  import logging
  import numpy as np
  import pypbip as pb

  # bookkeeping initialization
  logger = logging.getLogger(__name__)

  # algorithmic initialization
  u_patches = pb.seq2matrix(pb.image_to_vectors(u, pg))

  if int(K) != K:
    # use K*u.shape[1] 
    K = int(K * u_patches.shape[1])

  # create sparse representation of the noisy image
  [D, u_sparse_coeffs] = pb.ksvd(Y=u_patches, K=K, T=T,
      max_err=max_err, max_iter=max_iter, approx=ksvd_approx,
      preserve_dc=preserve_dc)
  u_sparse = np.dot(D, u_sparse_coeffs)
  u_sparse_seq = pb.column_gen(u_sparse)

  # reconcile various patches with one another and the data fidelity
  # term:
  num = alpha * u
  den = alpha * np.ones(u.shape)
  for (patch_coords, u_patch) in zip(pg, u_sparse_seq):
    # TODO consider more sophisticated patch weighting
    num_patch = pb.extract_patch(num, patch_coords)
    num_patch += u_patch.reshape(num_patch.shape)

    den_patch = pb.extract_patch(den, patch_coords)
    den_patch += 1

  assert(np.all(np.isfinite(num)))
  assert(np.all(np.isfinite(den)))
  assert(np.all(-np.isnan(num)))
  assert(np.all(-np.isnan(den)))

  to_return = num / den

  return to_return

