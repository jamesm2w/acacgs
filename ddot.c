#include "ddot.h"

#include <immintrin.h>
#include <omp.h>

/**
 * @brief Compute the dot product of two vectors
 *
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot(const int n, const double* const x, const double* const y, double* const result) {
	double local_result = 0.0;
	__m256d result_vec = _mm256_set1_pd(local_result);

	int loopN = (n / 4) * 4;
	int i = 0;

	if (y == x) {

		for (i = 0; i < loopN; i += 4) {
			__m256d xvec = _mm256_load_pd(x + i);       // Load x[1..4]
			__m256d xvec2 = _mm256_mul_pd(xvec, xvec);  // Multiply vector by vector
			result_vec = _mm256_add_pd(result_vec, xvec2);
		}

		// Cleanup loop
		for (; i < n; i++) {
			local_result += x[i] * x[i];
		}
	} else {

		for (i = 0; i < loopN; i += 4) {
			__m256d xvec = _mm256_load_pd(x + i);  // Load x[1..4]
			__m256d yvec = _mm256_load_pd(y + i);
			__m256d rvec = _mm256_mul_pd(xvec, yvec);  // Multiply vector by vector
			result_vec = _mm256_add_pd(result_vec, rvec);
		}

		for (; i < n; i++) {
			local_result += x[i] * y[i];
		}
	}

	__m128d result_high = _mm256_castpd256_pd128(result_vec); // Get bottom 128 bits (2 doubles)
	__m128d result_low = _mm256_extractf128_pd(result_vec, 1); // Get top 128 bits (2 doubles)
	result_low = _mm_add_pd(result_low, result_high); // Add the top to the bottom to get a0 + a3, a2 + a4?
	__m128d high64 = _mm_unpackhi_pd(result_low, result_low); // Get upper 64 bits (1 double)
	local_result += _mm_cvtsd_f64(_mm_add_sd(result_low, high64)); // Add top to bottom, cast out bottom as result

	*result = local_result;

	return 0;
}
