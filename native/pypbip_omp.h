#ifndef _PYPBIP_OMP_H_
#define _PYPBIP_OMP_H_

#include <stdbool.h>
#include <stdint.h>

/** \brief performs orthogonal matching pursuit on the given vector and
 * dictionary.
 *
 * Uses at most T atoms from the dictionary, and stops when the
 * representation error falls below the given value.
 *
 * Assumes data is in Fortran order, i.e., column-major.  THIS IS NOT
 * THE "STANDARD" C REPRESENTATION */
bool pypbip_omp_sf(
    int32_t N,
    const float *y,
    int32_t K,
    const float *D,
    float *x,
    int T,
    float err);

/** \brief performs othogonal matching pursuit on the given matrix of
 * vectors and dictionary.
 *
 * Uses at most T atoms from the dictionary for each training vector,
 * and stops when the representation error falls below the given
 * threshold.
 *
 * Assumes data is in Fortran order, i.e., column-major.  THIS IS NOT
 * THE "STANDARD" C REPRESENTATION */
bool pypbip_omp_batch(
    int32_t N,
    int32_t num_vec,
    float *y,
    int32_t K,
    float *D,
    float *X,
    int8_t T,
    float err);

#endif

