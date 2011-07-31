#include "pypbip_omp.h"

#include <cblas.h>
#include <clapack.h>

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT(what, label) \
  if(! (what) ) { \
    fprintf(stderr, "ASSERTION FAILED at line %d in file %s\n", \
        __LINE__, __FILE__); \
    goto label; \
  }

/** \brief workspace for running omp on a single vector */
typedef struct {
  float *Q;
  float *R;
  bool *used;
  int *used_indices;
  float *residual;
  float *new_atom;
  float *sparse_coeffs;
  float *Qt_y;
} omp_workspace;

static bool omp_workspace_init(omp_workspace *w,
    int32_t N, int32_t K, int T) {
  bool ok = true;

  float *Q = NULL;
  float *R = NULL;
  bool *used = NULL;
  int *used_indices = NULL;
  float *residual = NULL;
  float *new_atom = NULL;
  float *sparse_coeffs = NULL;
  float *Qt_y = NULL;

  /* TODO it would be nice to reduce all these callocs to one... */

  /* allocate space for Q, R, etc. */
  Q = calloc(sizeof(float), N * T);
  ASSERT(Q != NULL, fatal);

/* TODO i could use less memory in representing R if I was clever about
 * it... */
  R = calloc(sizeof(float), T * T);
  ASSERT(R != NULL, fatal);

  used = calloc(sizeof(bool), K);
  ASSERT(used != NULL, fatal);

  used_indices = calloc(sizeof(int), T);
  ASSERT(used_indices != NULL, fatal);

  residual = calloc(sizeof(float), N);
  ASSERT(residual != NULL, fatal);

  new_atom = calloc(sizeof(float), N);
  ASSERT(new_atom != NULL, fatal);

  sparse_coeffs = calloc(sizeof(float), T);
  ASSERT(sparse_coeffs != NULL, fatal);

  Qt_y = calloc(sizeof(float), T);
  ASSERT(Qt_y != NULL, fatal);

  /* copy stuff over, and mark each copy as successful (setting local
   * pointer to NULL) */
  w->Q = Q; Q = NULL;
  w->R = R; R = NULL;
  w->used = used; used = NULL;
  w->used_indices = used_indices; used_indices = NULL;
  w->residual = residual; residual = NULL;
  w->new_atom = new_atom; new_atom = NULL;
  w->sparse_coeffs = sparse_coeffs; sparse_coeffs = NULL;
  w->Qt_y = Qt_y; Qt_y = NULL;

  if(0) {
fatal:
    ok = false;

    /* erase memory allocated before failure */
    if(Q) free(Q);
    if(R) free(R);
    if(used) free(used);
    if(used_indices) free(used_indices);
    if(residual) free(residual);
    if(new_atom) free(new_atom);
    if(sparse_coeffs) free(sparse_coeffs);
    if(Qt_y) free(Qt_y);
  }

  return ok;
}

static bool omp_workspace_clear(omp_workspace *w,
    int32_t N, int32_t K, int T) {
  int i;
  memset(w->Q, 0, sizeof(float) * N * T);
  memset(w->R, 0, sizeof(float) * T * T);
  for(i=0; i<K; ++i) w->used[i] = false;
  memset(w->used_indices, 0, sizeof(int) * T);
  memset(w->residual, 0, sizeof(float) * N);
  memset(w->new_atom, 0, sizeof(float) * N);
  memset(w->sparse_coeffs, 0, sizeof(float) * T);
  memset(w->Qt_y, 0, sizeof(float) * T);
  return true;
}

static bool omp_workspace_del(omp_workspace *w) {
  free(w->Q); w->Q = NULL;
  free(w->R); w->R = NULL;
  free(w->used); w->used = NULL;
  free(w->used_indices); w->used_indices = NULL;
  free(w->residual); w->residual = NULL;
  free(w->new_atom); w->new_atom = NULL;
  free(w->sparse_coeffs); w->sparse_coeffs = NULL;
  free(w->Qt_y); w->Qt_y = NULL;
  return true;
}

/** \brief perform back substitution.
  Solves the problem R{out} = b subject to:
 - R is square a square TxT float matrix in column major, but we're
 interested in inverting only the NxN upper-left block. 
 - The NxN upper-left block is upper triangular
 
 NO CHECKS ARE MADE TO ENSURE THESE CONDITIONS ARE MET */
static bool backsub(int T, const float *R, const float *b, int N, float *out) {
  int i,j;
  float accum;

#define R_COL(c) (R+(c)*T)

  for(i=N-1; i>=0; --i) {
    accum = b[i];
    for(j=N-1; j>i; --j) {
      accum -= (R_COL(j)[i] * out[j]);
    }
    out[i] = accum / R_COL(i)[i];
  }

#undef R_COL

  /* i'm not checking for floating point faults here... */
  return true;
}

/* totally arbitrary floating-point epsilon */
#define EPS .000000001

static bool pypbip_omp_sf_do(
    int32_t N,
    const float *y,
    int32_t K,
    const float *D,
    float *x,
    int T,
    float err_thresh,
    const omp_workspace *w) {
  bool ok = true;

  float *Q = w->Q;
  float *R = w->R;
  bool *used = w->used;
  int *used_indices = w->used_indices;
  float *residual = w->residual;
  float *new_atom = w->new_atom;
  float *sparse_coeffs = w->sparse_coeffs;
  float *Qt_y = w->Qt_y;

  int8_t l0_norm = 0;
  float repr_err = cblas_sdot(N, y, 1, y, 1);
  
  float iprod;
  int i;

  memcpy(residual, y, sizeof(float)*N);

#define D_COL(i) (D+N*(i))
#define Q_COL(i) (Q+N*(i))
#define R_COL(i) (R+T*(i))

  /* main loop */
  while((l0_norm < T) && (repr_err > err_thresh)) {
    /* compute inner products and choose the maximum */
    float max_iprod = 0;
    float a_max_iprod = 0;
    int max_iprod_index = -1;
    for(i=0; i<K; ++i) {
      if(used[i]) continue;
      iprod = cblas_sdot(N, residual, 1, D_COL(i), 1);
      float a_iprod = fabsf(iprod);

      if(a_iprod > a_max_iprod) {
        max_iprod_index = i;
        a_max_iprod = a_iprod;
        max_iprod = iprod;
      }
    }
    if(max_iprod_index == -1) break;

    /* make a copy of the new atom; we'll be doing a lot of operations
     * on it */
    memcpy(new_atom, D_COL(max_iprod_index), N*sizeof(float));

    /* march through the columns of Q to orthogonalize the contribution
     * by the new atom */
    for(i=0; i<l0_norm; ++i) {
      iprod = cblas_sdot(N, new_atom, 1, Q_COL(i), 1);
      /* new_atom <- new_atom - iprod * Q_COL(i) */
      cblas_saxpy(N, -iprod, Q_COL(i), 1, new_atom, 1);
      /* update R */
      R_COL(l0_norm)[i] = iprod;
    }

    /* mark the selected inner product as used */
    used[max_iprod_index] = true;

    /* check if we've blown away the new atom... i.e., we selected two
     * linearly dependent atoms... oops. */
    float new_atom_norm2 = cblas_sdot(N, new_atom, 1, new_atom, 1);
    if(new_atom_norm2 < EPS) {
      /* okay, we made a mistake.  we'll mark this atom as used (so we
       * won't try to use it again) but won't increase the l0 norm of
       * our approximation.  back to the top. */
      continue;
    }

    /* normalize the new atom */
    float new_atom_norm = sqrtf(new_atom_norm2);
    for(i=0; i<N; ++i) new_atom[i] /= new_atom_norm;

    /* set the diagonal element of R */
    R_COL(l0_norm)[l0_norm] = new_atom_norm;

    /* indicate that we're using this atom in the reconstruction */
    used_indices[l0_norm] = max_iprod_index;

    /* update the residual */
    iprod = cblas_sdot(N, residual, 1, new_atom, 1);
    cblas_saxpy(N, -iprod, new_atom, 1, residual, 1);

    /* update representation error */
    /* TODO this could be expressed in terms of values already computed,
     * but we'll keep it this way for now */
    repr_err = sqrtf( cblas_sdot(N, residual, 1, residual, 1) );

    /* store this inner product for reconstruction later */
    Qt_y[l0_norm] = iprod;

    /* add the new atom to Q */
    memcpy(Q_COL(l0_norm), new_atom, sizeof(float)*N);

    /* we've added one more nonzero coefficient to x... */
    ++l0_norm;
  }

  /* we've computed QRx = Dx = y, and Q'y.  all that remains is to find
   * R^{-1}Q'y, which can be done via (relatively) simple
   * back-substitution.  i couldn't find a LAPACK routine that did what
   * i wanted, so we're using a home grown solution :/ */
  ASSERT( backsub(T, R, Qt_y, l0_norm, sparse_coeffs), fatal );

  /* now, expand our sparse representation of x */
  memset(x, 0, sizeof(float)*K);
  for(i=0; i<l0_norm; ++i) x[used_indices[i]] = sparse_coeffs[i];

  if(0) {
fatal:
    ok = false;

    goto cleanup;
  }

cleanup:
  return ok;
}

bool pypbip_omp_sf(
    int32_t N,
    const float *y,
    int32_t K,
    const float *D,
    float *x,
    int T,
    float err_thresh) {
  bool ok = true;
  omp_workspace w;

  ASSERT(omp_workspace_init(&w, N, K, T), fatal);
  ASSERT(pypbip_omp_sf_do(N, y, K, D, x, T, err_thresh, &w), fatal);

  if(0) {
fatal:
    ok = false;
  }

  omp_workspace_del(&w);
  return ok;
}

bool pypbip_omp_batch_sf(
    int32_t N,
    int32_t M,
    float *y,
    int32_t K,
    float *D,
    float *X,
    int T,
    float err) {
  /* FIXME currently, while the batch run is awesome and multithreaded,
   * it doesn't report errors correctly. */
  omp_workspace w;
  bool ok = true;
  bool *runs_ok = NULL;
  int i;

  /* allocate space for error checks */
  runs_ok = malloc(sizeof(bool) * M);
  ASSERT(runs_ok != NULL, fatal);
  for(i=0; i<M; ++i) runs_ok[i] = true;

#pragma omp parallel shared(y,D,X,runs_ok) private(w)
  {
    /* per-thread workspace */
    omp_workspace_init(&w, N, K, T);

    /* split across threads */
#pragma omp for schedule(static) nowait
    for(i=0; i<M; ++i) {
      omp_workspace_clear(&w, N, K, T);
      runs_ok[i] = pypbip_omp_sf_do(N, y+N*i, K, D, X+K*i, T, err, &w);
    }

    omp_workspace_del(&w);
  } /* end of parallel section */

  /* find if any threads failed */
  for(i=0; i<M; ++i) ok &= runs_ok[i];

  if(0) {
fatal:
    ok = false;
  }

  if(runs_ok) free(runs_ok);
  return ok;
}

/* eof */

