def ksvd_constrained_denoise(alpha, u, pg, T, err=0, max_iter=10):
  """Attempts to denoise the image u using ksvd and some sort of
  "sparsity prior".

  This implements something close to the denoising algorithm described
  in Michael Elad and Michal Aharon's 2006 paper "Image Denoising via
  Sparse and Redundant Representations over Learned Dictionaries" using
  the noisy image as a prior.

  As per the paper, it's _very important_ to set T, the maximum sparse
  coefficient l0-norm and error according to the estimated noise level
  of the image.  In the paper mentioned above, setting err = 1.15 * sigma
  is recommended, and lamb = 30/sigma ... assuming AWGN.

  """
  import logging
  import numpy as np
  import pypbip as pb

  # bookkeeping initialization
  logger = logging.getLogger(__name__)

  # algorithmic initialization
  u_patches = pb.seq2matrix(image_to_vectors(z, pg))
  K = u_patches.size[1]

  # create sparse representation of the noisy image
  [D, u_sparse_coeffs] = pb.ksvd(Y=u_patches, K=K, T=T,
      max_err=err, max_iter=max_iter)
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

  return num_patch / den_patch

