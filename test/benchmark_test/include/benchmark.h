#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// A simple C struct representing a strided 2D array view.
// This closely matches what Julia's multi-dimensional arrays look like in memory.
typedef struct StridedMatrixView {
    double* data;
    size_t rows;
    size_t cols;
    size_t stride_row;
    size_t stride_col;
} StridedMatrixView;

// Multiplies two matrices: C = A * B
// Operates directly on the strided views without copying.
void multiply_matrices(const StridedMatrixView* A, const StridedMatrixView* B, StridedMatrixView* C);

// Simple contiguous array processing for baseline comparison
void process_contiguous_array(double* data, size_t length, double multiplier);

#ifdef __cplusplus
}
#endif

#endif // BENCHMARK_H