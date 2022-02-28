#include "waxpby.h"

#include <immintrin.h>

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */
int waxpby (const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w) {  
  
  if (alpha == 1.0) {

    //  #pragma GCC unroll 4
    __m256d bvec = _mm256_set1_pd(beta);

    int loopN = (n / 4) * 4;
    int i = 0;
    for (i = 0; i < loopN; i += 4) {

      // __m256d xvec = _mm256_load_pd(x + i);
      // __m256d yvec = _mm256_load_pd(y + i);
      // yvec = _mm256_mul_pd(yvec, bvec);

      _mm256_store_pd(w + i, _mm256_add_pd(_mm256_load_pd(x + i), _mm256_mul_pd(_mm256_load_pd(y + i), bvec)));
    }
    for (; i < n; i++) {
      w[i] = x[i] + beta * y[i];
    }

  } else if (beta == 1.0) {

    // #pragma GCC unroll 4
    __m256d avec = _mm256_set1_pd(alpha);

    int loopN = (n / 4) * 4;
    int i = 0;
    for (i = 0; i < loopN; i += 4) {

//       __m256d xvec = _mm256_load_pd(x + i);
//       __m256d yvec = _mm256_load_pd(y + i);
//       xvec = _mm256_mul_pd(xvec, avec);

      _mm256_store_pd(w + i, _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(x + i), avec), _mm256_load_pd(y + i)));
    }

    for (; i < n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  
  } else {
  
    // #pragma GCC unroll 4
    __m256d avec = _mm256_set1_pd(alpha);
    __m256d bvec = _mm256_set1_pd(beta);

    int loopN = (n / 4) * 4;
    int i = 0;
    for (i = 0; i < loopN; i += 4) {

      //__m256d xvec = _mm256_load_pd(x + i);
      //__m256d yvec = _mm256_load_pd(y + i);
      //xvec = _mm256_mul_pd(xvec, avec);
      //yvec = _mm256_mul_pd(yvec, bvec);

      _mm256_store_pd(w + i, _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(x + i), avec), _mm256_mul_pd(_mm256_load_pd(y + i), bvec)));
    }

    for (; i < n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  
  }

  return 0;
}
