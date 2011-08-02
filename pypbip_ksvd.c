#include "pypbip_ksvd.h"
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

bool pypbip_ksvd(
    int N,
    int M,
    float *Y,
    int K,
    float *D,
    float *X,
    float max_err,
    int max_iter) {
  bool ok = true;

  if(0) {
fatal:
    ok = false;
  }

  return ok;
}

/* eof */

