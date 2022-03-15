#include "waxpby.h"

#include <immintrin.h>

#include <stdio.h>

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
int waxpby(const int n, const double* const x, const double beta, const double* const y, double* const w) {
	// alpha is always 1.0 -- Only the alpha == 1 branch is necessary
	int loopN = (n / 4) * 4;
	int i = 0;

	if (beta == 0.0) {
		// Equivalent to x + 0 * y, i.e. just x

		for (i = 0; i < loopN; i += 4) {
			w[i] = x[i];
			w[i+1] = x[i+1];
			w[i+2] = x[i+2];
			w[i+3] = x[i+3];
		}

		for (; i < n; i++) {
			w[i] = x[i];
		}

	} else if (beta == 1.0) {
		// Equivalent to an add of x + y
			
		for (i = 0; i < loopN; i += 4) {
			__m256d x_vec = _mm256_load_pd(x + i);
			__m256d y_vec = _mm256_load_pd(y + i);

			x_vec = _mm256_add_pd(x_vec, y_vec);

			_mm256_store_pd(w + i, x_vec);
		}

		for (; i < n; i++) {
			w[i] = x[i] + y[i];
		}
	} else {
		// beta is not 0 or 1 and we have to do the calculation
		__m256d b_vec = _mm256_set1_pd(beta);
		
		for (i = 0; i < loopN; i += 4) {
			__m256d x_vec = _mm256_load_pd(x + i);
			__m256d y_vec = _mm256_load_pd(y + i);

			y_vec = _mm256_mul_pd(y_vec, b_vec);
			x_vec = _mm256_add_pd(x_vec, y_vec);

			_mm256_store_pd(w + i, x_vec);
		}

		for (; i < n; i++) {
			w[i] = x[i] + beta * y[i];
		}
	}

	return 0;
}
