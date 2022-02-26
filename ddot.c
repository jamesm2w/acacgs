#include "ddot.h"

#include <omp.h>
#include <immintrin.h>

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const double * const x, const double * const y, double * const result) {  
  double local_result = 0.0;
  
  if (y==x){
    
    //#pragma GCC unroll 4
    
    int loopN = (n / 4) * 4;
    int i = 0;
    // #pragma omp parallel for reduction(+:local_result) private(i)
    for (i = 0; i < loopN; i += 4) {
      __m256d xvec = _mm256_load_pd(x + i); // Load x[1..4]
      __m256d xvec2 = _mm256_mul_pd(xvec, xvec); // Multiply vector by vector

      __m128d xlow = _mm256_castpd256_pd128(xvec2); // Get bottom 128 bits (2 doubles)
      __m128d xhigh = _mm256_extractf128_pd(xvec2, 1); // Get top 128 bits (2 doubles)
      xlow = _mm_add_pd(xlow, xhigh); // Add the top to the bottom to get a0 + a3, a2 + a4?

      __m128d high64 = _mm_unpackhi_pd(xlow, xlow); // Get upper 64 bits (1 double)

      local_result += _mm_cvtsd_f64(_mm_add_sd(xlow, high64)); // Add top to bottom, cast out bottom as result
      //local_result += x[i] * x[i];
    }

    for (; i < n; i++) {
      local_result += x[i] * x[i];
    }

  } else {
    //#pragma omp parallel for reduction(+:local_result)
    // #pragma GCC unroll 4
    // for (int i=0; i<n; i++) {
    //   local_result += x[i]*y[i];
    // }

    int loopN = (n / 4) * 4;
    int i = 0;
    // #pragma omp parallel for reduction(+:local_result) private(i) - Slows down code, like by a lot :lawl:
    for (i = 0; i < loopN; i += 4) {
      __m256d xvec = _mm256_load_pd(x + i); // Load x[1..4]
      __m256d yvec = _mm256_load_pd(y + i);
      __m256d rvec = _mm256_mul_pd(xvec, yvec); // Multiply vector by vector

      __m128d rlow = _mm256_castpd256_pd128(rvec); // Get bottom 128 bits (2 doubles)
      __m128d rhigh = _mm256_extractf128_pd(rvec, 1); // Get top 128 bits (2 doubles)
      rlow = _mm_add_pd(rlow, rhigh); // Add the top to the bottom to get a0 + a3, a2 + a4?

      __m128d high64 = _mm_unpackhi_pd(rlow, rlow); // Get upper 64 bits (1 double)

      local_result += _mm_cvtsd_f64(_mm_add_sd(rlow, high64)); // Add top to bottom, cast out bottom as result
      //local_result += x[i] * x[i];
    }

    for (; i < n; i++) {
      local_result += x[i] * y[i];
    }
  }
  *result = local_result;

  return 0;
}
