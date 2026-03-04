#include "benchmark.h"

void multiply_matrices(const StridedMatrixView* A, const StridedMatrixView* B, StridedMatrixView* C) {
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        return; // Dimension mismatch
    }

    for (size_t i = 0; i < A->rows; ++i) {
        for (size_t j = 0; j < B->cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A->cols; ++k) {
                // Manually compute strided offsets
                double a_val = A->data[i * A->stride_row + k * A->stride_col];
                double b_val = B->data[k * B->stride_row + j * B->stride_col];
                sum += a_val * b_val;
            }
            C->data[i * C->stride_row + j * C->stride_col] = sum;
        }
    }
}

void process_contiguous_array(double* data, size_t length, double multiplier) {
    for (size_t i = 0; i < length; ++i) {
        data[i] *= multiplier;
    }
}