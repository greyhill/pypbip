#ifndef _PYPBIP_KSVD_H_
#define _PYPBIP_KSVD_H_

#include <stdbool.h>
#include <stdint.h>

/** \brief performs the k-svd dictionary learning / sparse
 * representation algorithm.  hopefully, it does this quickly. */
bool pypbip_ksvd(
    int N,
    int M,
    float *Y,
    int K,
    float *D,
    float *X,
    float max_err,
    int max_iter);

#endif /* eof */

