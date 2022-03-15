#include "sparsemv.h"

#include <immintrin.h>
#include <omp.h>
/**
 * @brief Compute matrix vector product (y = A*x)
 *
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const double *const x, double *const y) {
	const int nrow = (const int)A->local_nrow;  // number of rows in A
	int i = 0;

	#pragma omp parallel for
	for (i = 0; i < nrow; i++) {
		double sum = 0.0;
		__m256d sum_vec = _mm256_set1_pd(sum);

		const double *const cur_vals = (const double *const)A->ptr_to_vals_in_row[i];  // double array of values in row i of A
		const int *const cur_inds = (const int *const)A->ptr_to_inds_in_row[i];        // int array of column indexes of non-zero values in row i
		const int cur_nnz = (const int)A->nnz_in_row[i];                               // int number of non-zero numbers in row i

		int loopN = (cur_nnz / 4) * 4;
		int j;
		for (j = 0; j < loopN; j += 4) {  // Loops through all non-zeros in the row `i`

			__m256d cur_vals_j = _mm256_loadu_pd(cur_vals + j);  // non-zero values in row `i` of the matrix

			const __m128i *cur_inds_2 = (const __m128i *)(cur_inds + j);
			__m128i cur_inds_reg = _mm_loadu_si128(cur_inds_2);              // load the column offsets of the non-zero values -- there is no performance difference on skylake for this vs aligned load :woozy_face:
			__m256d x_cur_inds_j = _mm256_i32gather_pd(x, cur_inds_reg, 8);  // gather them. 8 means count in bytes

			sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(cur_vals_j, x_cur_inds_j));  // Accumulate into sum_vec
		}

		// Reduce sum vector to just one value by horizontal add
		__m128d sum_vec_low = _mm256_castpd256_pd128(sum_vec);     // Low  128 bits
		__m128d sum_vec_high = _mm256_extractf128_pd(sum_vec, 1);  // High 128 bits
		sum_vec_low = _mm_add_pd(sum_vec_low, sum_vec_high);       // Add top to bottom
		__m128d high64 = _mm_unpackhi_pd(sum_vec_low, sum_vec_low);

		sum += _mm_cvtsd_f64(_mm_add_sd(sum_vec_low, high64));

		// Cleanup loop
		for (; j < cur_nnz; j++) {
			sum = sum + cur_vals[j] * x[cur_inds[j]];
		}

		y[i] = sum;
	}

	// for (i = 0; i < nrow; i++) {
	//   double sum = 0.0;

	//   const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
	//   const int    * const cur_inds = (const int    * const) A->ptr_to_inds_in_row[i];
	//   const int             cur_nnz = (const int)            A->nnz_in_row[i];

	//   #pragma GCC unroll(4)
	//   for (int j = 0; j < cur_nnz; j++) {
	//     sum = sum + cur_vals[j] * x[cur_inds[j]];
	//   }

	//   y[i] = sum;
	// }

	return 0;
}
