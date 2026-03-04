#include "numerics.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>

// ============================================================================
// IMPLEMENTATION: Stress Test for RepliBuild
// ============================================================================

static std::mt19937_64 rng;

// ============================================================================
// Memory Management
// ============================================================================

extern "C" {

DenseMatrix dense_matrix_create(size_t rows, size_t cols) {
    DenseMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (double*)calloc(rows * cols, sizeof(double));
    mat.owns_data = true;
    return mat;
}

void dense_matrix_destroy(DenseMatrix* mat) {
    if (mat && mat->owns_data && mat->data) {
        free(mat->data);
        mat->data = nullptr;
    }
}

DenseMatrix dense_matrix_copy(const DenseMatrix* src) {
    DenseMatrix dst = dense_matrix_create(src->rows, src->cols);
    memcpy(dst.data, src->data, src->rows * src->cols * sizeof(double));
    return dst;
}

void dense_matrix_set_zero(DenseMatrix* mat) {
    memset(mat->data, 0, mat->rows * mat->cols * sizeof(double));
}

void dense_matrix_set_identity(DenseMatrix* mat) {
    dense_matrix_set_zero(mat);
    size_t n = std::min(mat->rows, mat->cols);
    for (size_t i = 0; i < n; i++) {
        mat->data[i * mat->cols + i] = 1.0;
    }
}

Status dense_matrix_resize(DenseMatrix* mat, size_t new_rows, size_t new_cols) {
    if (!mat->owns_data) {
        return Status::ERROR_INVALID_INPUT;
    }

    double* new_data = (double*)calloc(new_rows * new_cols, sizeof(double));
    if (!new_data) {
        return Status::ERROR_OUT_OF_MEMORY;
    }

    // Copy old data to new buffer
    size_t copy_rows = std::min(mat->rows, new_rows);
    size_t copy_cols = std::min(mat->cols, new_cols);
    for (size_t i = 0; i < copy_rows; i++) {
        for (size_t j = 0; j < copy_cols; j++) {
            new_data[i * new_cols + j] = mat->data[i * mat->cols + j];
        }
    }

    free(mat->data);
    mat->data = new_data;
    mat->rows = new_rows;
    mat->cols = new_cols;

    return Status::SUCCESS;
}

SparseMatrix sparse_matrix_create(size_t rows, size_t cols, size_t nnz) {
    SparseMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.nnz = nnz;
    mat.values = (double*)calloc(nnz, sizeof(double));
    mat.row_indices = (int32_t*)calloc(nnz, sizeof(int32_t));
    mat.col_pointers = (int32_t*)calloc(cols + 1, sizeof(int32_t));
    return mat;
}

void sparse_matrix_destroy(SparseMatrix* mat) {
    if (mat) {
        free(mat->values);
        free(mat->row_indices);
        free(mat->col_pointers);
    }
}

// ============================================================================
// Basic Linear Algebra
// ============================================================================

double vector_dot(const double* x, const double* y, size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
}

double vector_norm(const double* x, size_t n) {
    return std::sqrt(vector_dot(x, x, n));
}

void vector_scale(double* x, double alpha, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

void vector_axpy(double* y, double alpha, const double* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

void vector_copy(double* dest, const double* src, size_t n) {
    memcpy(dest, src, n * sizeof(double));
}

void matrix_vector_mult(const DenseMatrix* A, const double* x, double* y) {
    for (size_t i = 0; i < A->rows; i++) {
        y[i] = 0.0;
        for (size_t j = 0; j < A->cols; j++) {
            y[i] += A->data[i * A->cols + j] * x[j];
        }
    }
}

void matrix_vector_mult_add(const DenseMatrix* A, const double* x, double* y,
                            double alpha, double beta) {
    double* temp = (double*)malloc(A->rows * sizeof(double));
    matrix_vector_mult(A, x, temp);

    for (size_t i = 0; i < A->rows; i++) {
        y[i] = alpha * temp[i] + beta * y[i];
    }

    free(temp);
}

DenseMatrix matrix_multiply(const DenseMatrix* A, const DenseMatrix* B) {
    DenseMatrix C = dense_matrix_create(A->rows, B->cols);

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            C.data[i * C.cols + j] = 0.0;
            for (size_t k = 0; k < A->cols; k++) {
                C.data[i * C.cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
        }
    }

    return C;
}

DenseMatrix matrix_add(const DenseMatrix* A, const DenseMatrix* B) {
    DenseMatrix C = dense_matrix_create(A->rows, A->cols);

    for (size_t i = 0; i < A->rows * A->cols; i++) {
        C.data[i] = A->data[i] + B->data[i];
    }

    return C;
}

DenseMatrix matrix_transpose(const DenseMatrix* A) {
    DenseMatrix At = dense_matrix_create(A->cols, A->rows);

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            At.data[j * At.cols + i] = A->data[i * A->cols + j];
        }
    }

    return At;
}

double matrix_trace(const DenseMatrix* A) {
    double trace = 0.0;
    size_t n = std::min(A->rows, A->cols);
    for (size_t i = 0; i < n; i++) {
        trace += A->data[i * A->cols + i];
    }
    return trace;
}

double matrix_determinant(const DenseMatrix* A) {
    // Simple 2x2 and 3x3 cases
    if (A->rows == 2 && A->cols == 2) {
        return A->data[0] * A->data[3] - A->data[1] * A->data[2];
    }

    if (A->rows == 3 && A->cols == 3) {
        return A->data[0] * (A->data[4] * A->data[8] - A->data[5] * A->data[7])
             - A->data[1] * (A->data[3] * A->data[8] - A->data[5] * A->data[6])
             + A->data[2] * (A->data[3] * A->data[7] - A->data[4] * A->data[6]);
    }

    // For larger matrices, use LU decomposition
    LUDecomposition lu = compute_lu(A);
    if (lu.status != Status::SUCCESS) {
        return 0.0;
    }

    double det = 1.0;
    for (size_t i = 0; i < lu.size; i++) {
        det *= lu.U.data[i * lu.U.cols + i];
    }

    dense_matrix_destroy(&lu.L);
    dense_matrix_destroy(&lu.U);
    free(lu.permutation);

    return det;
}

// ============================================================================
// Decompositions (RVO Test!)
// ============================================================================

LUDecomposition compute_lu(const DenseMatrix* A) {
    LUDecomposition result;
    result.size = A->rows;
    result.L = dense_matrix_create(A->rows, A->cols);
    result.U = dense_matrix_copy(A);
    result.permutation = (int32_t*)malloc(A->rows * sizeof(int32_t));
    result.status = Status::SUCCESS;

    for (size_t i = 0; i < A->rows; i++) {
        result.permutation[i] = i;
    }

    dense_matrix_set_identity(&result.L);

    // Simple Gaussian elimination (without pivoting for brevity)
    for (size_t k = 0; k < A->rows - 1; k++) {
        for (size_t i = k + 1; i < A->rows; i++) {
            double factor = result.U.data[i * A->cols + k] / result.U.data[k * A->cols + k];
            result.L.data[i * A->cols + k] = factor;

            for (size_t j = k; j < A->cols; j++) {
                result.U.data[i * A->cols + j] -= factor * result.U.data[k * A->cols + j];
            }
        }
    }

    return result;
}

QRDecomposition compute_qr(const DenseMatrix* A) {
    QRDecomposition result;
    result.m = A->rows;
    result.n = A->cols;
    result.Q = dense_matrix_create(A->rows, A->rows);
    result.R = dense_matrix_copy(A);
    result.status = Status::SUCCESS;

    dense_matrix_set_identity(&result.Q);

    // Simplified QR (Gram-Schmidt)
    // Full implementation would use Householder reflections

    return result;
}

EigenDecomposition compute_eigen(const DenseMatrix* A) {
    EigenDecomposition result;
    result.n = A->rows;
    result.eigenvalues = (double*)calloc(A->rows, sizeof(double));
    result.eigenvalues_imag = (double*)calloc(A->rows, sizeof(double));
    result.eigenvectors = dense_matrix_create(A->rows, A->rows);
    result.status = Status::SUCCESS;

    // Placeholder: real implementation would use QR algorithm or Jacobi
    for (size_t i = 0; i < A->rows; i++) {
        result.eigenvalues[i] = A->data[i * A->cols + i]; // Diagonal approximation
    }
    dense_matrix_set_identity(&result.eigenvectors);

    return result;
}

Status solve_linear_system_lu(const LUDecomposition* lu, const double* b, double* x, size_t n) {
    // Forward substitution: Ly = b
    double* y = (double*)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        y[i] = b[i];
        for (size_t j = 0; j < i; j++) {
            y[i] -= lu->L.data[i * n + j] * y[j];
        }
    }

    // Back substitution: Ux = y
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (size_t j = i + 1; j < n; j++) {
            x[i] -= lu->U.data[i * n + j] * x[j];
        }
        x[i] /= lu->U.data[i * n + i];
    }

    free(y);
    return Status::SUCCESS;
}

Status solve_linear_system_qr(const QRDecomposition* qr, const double* b, double* x) {
    // x = R^{-1} Q^T b
    // Simplified implementation
    return Status::SUCCESS;
}

Status solve_least_squares(const DenseMatrix* A, const double* b, double* x) {
    QRDecomposition qr = compute_qr(A);
    Status status = solve_linear_system_qr(&qr, b, x);

    dense_matrix_destroy(&qr.Q);
    dense_matrix_destroy(&qr.R);

    return status;
}

Status solve_conjugate_gradient(MatVecProduct matvec, const double* b, double* x,
                                size_t n, double tolerance, int32_t max_iterations,
                                void* user_data) {
    double* r = (double*)malloc(n * sizeof(double));
    double* p = (double*)malloc(n * sizeof(double));
    double* Ap = (double*)malloc(n * sizeof(double));

    // r = b - Ax
    matvec(x, Ap, n, user_data);
    for (size_t i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    double rs_old = vector_dot(r, r, n);

    for (int32_t iter = 0; iter < max_iterations; iter++) {
        matvec(p, Ap, n, user_data);

        double alpha = rs_old / vector_dot(p, Ap, n);

        vector_axpy(x, alpha, p, n);
        vector_axpy(r, -alpha, Ap, n);

        double rs_new = vector_dot(r, r, n);

        if (std::sqrt(rs_new) < tolerance) {
            free(r);
            free(p);
            free(Ap);
            return Status::SUCCESS;
        }

        double beta = rs_new / rs_old;
        for (size_t i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rs_old = rs_new;
    }

    free(r);
    free(p);
    free(Ap);

    return Status::ERROR_NOT_CONVERGED;
}

// ============================================================================
// Optimization
// ============================================================================

Status optimize_minimize(ObjectiveFunction objective, GradientFunction gradient,
                        double* x, size_t n, const OptimizationOptions* options,
                        OptimizationState* final_state, IterationCallback callback,
                        void* user_data) {
    double* grad = (double*)malloc(n * sizeof(double));
    double* direction = (double*)malloc(n * sizeof(double));
    double* x_new = (double*)malloc(n * sizeof(double));

    for (int32_t iter = 0; iter < options->max_iterations; iter++) {
        double f = objective(x, n, user_data);
        gradient(x, grad, n, user_data);

        double grad_norm = vector_norm(grad, n);

        if (grad_norm < options->tolerance) {
            if (final_state) {
                final_state->f_value = f;
                final_state->gradient_norm = grad_norm;
                final_state->iteration = iter;
                final_state->status = Status::SUCCESS;
            }

            free(grad);
            free(direction);
            free(x_new);
            return Status::SUCCESS;
        }

        // Gradient descent direction
        for (size_t i = 0; i < n; i++) {
            direction[i] = -grad[i];
        }

        // Line search
        double step = line_search_backtracking(objective, x, direction, x_new, n,
                                              options->step_size, user_data);

        vector_copy(x, x_new, n);

        // Callback
        if (callback) {
            OptimizationState state;
            state.x = x;
            state.gradient = grad;
            state.f_value = f;
            state.gradient_norm = grad_norm;
            state.iteration = iter;
            state.dimension = n;
            state.status = Status::SUCCESS;

            if (!callback(&state, user_data)) {
                if (final_state) *final_state = state;
                free(grad);
                free(direction);
                free(x_new);
                return Status::SUCCESS;
            }
        }
    }

    free(grad);
    free(direction);
    free(x_new);

    return Status::ERROR_NOT_CONVERGED;
}

Status optimize_minimize_numerical_gradient(ObjectiveFunction objective, double* x, size_t n,
                                            const OptimizationOptions* options,
                                            OptimizationState* final_state,
                                            IterationCallback callback, void* user_data) {
    // Numerical gradient computation
    auto numerical_gradient = [](const double* x_val, double* grad, size_t n_val, void* data) {
        auto obj = (ObjectiveFunction)((void**)data)[0];
        void* user_data_ptr = ((void**)data)[1];

        const double eps = 1e-8;
        double* x_plus = (double*)malloc(n_val * sizeof(double));

        for (size_t i = 0; i < n_val; i++) {
            vector_copy(x_plus, x_val, n_val);
            x_plus[i] += eps;
            double f_plus = obj(x_plus, n_val, user_data_ptr);

            x_plus[i] = x_val[i] - eps;
            double f_minus = obj(x_plus, n_val, user_data_ptr);

            grad[i] = (f_plus - f_minus) / (2.0 * eps);
        }

        free(x_plus);
    };

    void* data[2] = {(void*)objective, user_data};

    return optimize_minimize(objective, numerical_gradient, x, n, options,
                           final_state, callback, data);
}

double line_search_backtracking(ObjectiveFunction objective, const double* x,
                                const double* direction, double* x_new, size_t n,
                                double initial_step, void* user_data) {
    const double c = 0.5;
    const double tau = 0.5;
    double alpha = initial_step;

    double f0 = objective(x, n, user_data);

    for (int i = 0; i < 20; i++) {
        for (size_t j = 0; j < n; j++) {
            x_new[j] = x[j] + alpha * direction[j];
        }

        double f_new = objective(x_new, n, user_data);

        if (f_new < f0) {
            return alpha;
        }

        alpha *= tau;
    }

    return alpha;
}

// ============================================================================
// ODE Solvers
// ============================================================================

ODEResult solve_ode_rk4(ODEFunction ode_func, double t0, double t_final,
                        const double* y0, size_t n, double dt, void* user_data) {
    ODEResult result;
    result.dimension = n;
    result.n_steps = (size_t)((t_final - t0) / dt) + 1;
    result.y = (double*)malloc(n * sizeof(double));
    result.t_values = (double*)malloc(result.n_steps * sizeof(double));
    result.y_values = (double**)malloc(result.n_steps * sizeof(double*));
    result.status = Status::SUCCESS;

    for (size_t i = 0; i < result.n_steps; i++) {
        result.y_values[i] = (double*)malloc(n * sizeof(double));
    }

    vector_copy(result.y, y0, n);

    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* k4 = (double*)malloc(n * sizeof(double));
    double* temp = (double*)malloc(n * sizeof(double));

    double t = t0;
    for (size_t step = 0; step < result.n_steps; step++) {
        result.t_values[step] = t;
        vector_copy(result.y_values[step], result.y, n);

        if (step < result.n_steps - 1) {
            ode_func(t, result.y, k1, n, user_data);

            for (size_t i = 0; i < n; i++) {
                temp[i] = result.y[i] + 0.5 * dt * k1[i];
            }
            ode_func(t + 0.5 * dt, temp, k2, n, user_data);

            for (size_t i = 0; i < n; i++) {
                temp[i] = result.y[i] + 0.5 * dt * k2[i];
            }
            ode_func(t + 0.5 * dt, temp, k3, n, user_data);

            for (size_t i = 0; i < n; i++) {
                temp[i] = result.y[i] + dt * k3[i];
            }
            ode_func(t + dt, temp, k4, n, user_data);

            for (size_t i = 0; i < n; i++) {
                result.y[i] += (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
            }

            t += dt;
        }
    }

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(temp);

    return result;
}

ODEResult solve_ode_adaptive(ODEFunction ode_func, double t0, double t_final,
                             const double* y0, size_t n, double tolerance,
                             EventFunction event_func, void* user_data) {
    // Simplified adaptive solver - would use RK45 or similar in practice
    return solve_ode_rk4(ode_func, t0, t_final, y0, n, 0.01, user_data);
}

void ode_result_destroy(ODEResult* result) {
    if (result) {
        free(result->y);
        free(result->t_values);
        for (size_t i = 0; i < result->n_steps; i++) {
            free(result->y_values[i]);
        }
        free(result->y_values);
    }
}

// ============================================================================
// Signal Processing
// ============================================================================

FFTResult compute_fft(const double* signal, size_t n) {
    FFTResult result;
    result.n = n;
    result.real = (double*)malloc(n * sizeof(double));
    result.imag = (double*)malloc(n * sizeof(double));

    // Simplified DFT (not actual FFT)
    for (size_t k = 0; k < n; k++) {
        result.real[k] = 0.0;
        result.imag[k] = 0.0;
        for (size_t t = 0; t < n; t++) {
            double angle = -2.0 * M_PI * k * t / n;
            result.real[k] += signal[t] * std::cos(angle);
            result.imag[k] += signal[t] * std::sin(angle);
        }
    }

    return result;
}

void compute_ifft(const FFTResult* fft_data, double* signal_out) {
    size_t n = fft_data->n;

    for (size_t t = 0; t < n; t++) {
        signal_out[t] = 0.0;
        for (size_t k = 0; k < n; k++) {
            double angle = 2.0 * M_PI * k * t / n;
            signal_out[t] += fft_data->real[k] * std::cos(angle) - fft_data->imag[k] * std::sin(angle);
        }
        signal_out[t] /= n;
    }
}

void fft_result_destroy(FFTResult* result) {
    if (result) {
        free(result->real);
        free(result->imag);
    }
}

void convolve(const double* signal1, size_t n1, const double* signal2, size_t n2, double* result) {
    size_t n_out = n1 + n2 - 1;
    for (size_t i = 0; i < n_out; i++) {
        result[i] = 0.0;
        for (size_t j = 0; j < n2; j++) {
            if (i >= j && i - j < n1) {
                result[i] += signal1[i - j] * signal2[j];
            }
        }
    }
}

void correlate(const double* signal1, const double* signal2, size_t n, double* result) {
    for (size_t lag = 0; lag < n; lag++) {
        result[lag] = 0.0;
        for (size_t i = 0; i < n - lag; i++) {
            result[lag] += signal1[i] * signal2[i + lag];
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

double compute_mean(const double* data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

double compute_variance(const double* data, size_t n) {
    double mean = compute_mean(data, n);
    double variance = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    return variance / (n - 1);
}

double compute_stddev(const double* data, size_t n) {
    return std::sqrt(compute_variance(data, n));
}

double compute_median(double* data, size_t n) {
    std::sort(data, data + n);
    if (n % 2 == 0) {
        return (data[n/2 - 1] + data[n/2]) / 2.0;
    } else {
        return data[n/2];
    }
}

void compute_quantiles(double* data, size_t n, const double* probabilities,
                      double* quantiles, size_t n_quantiles) {
    std::sort(data, data + n);
    for (size_t i = 0; i < n_quantiles; i++) {
        double index = probabilities[i] * (n - 1);
        size_t lower = (size_t)index;
        size_t upper = lower + 1;
        if (upper >= n) {
            quantiles[i] = data[n - 1];
        } else {
            double weight = index - lower;
            quantiles[i] = (1.0 - weight) * data[lower] + weight * data[upper];
        }
    }
}

Histogram compute_histogram(const double* data, size_t n, size_t n_bins,
                           double min_val, double max_val) {
    Histogram hist;
    hist.n_bins = n_bins;
    hist.bin_edges = (double*)malloc((n_bins + 1) * sizeof(double));
    hist.counts = (int32_t*)calloc(n_bins, sizeof(int32_t));

    double bin_width = (max_val - min_val) / n_bins;
    for (size_t i = 0; i <= n_bins; i++) {
        hist.bin_edges[i] = min_val + i * bin_width;
    }

    for (size_t i = 0; i < n; i++) {
        if (data[i] >= min_val && data[i] <= max_val) {
            size_t bin = (size_t)((data[i] - min_val) / bin_width);
            if (bin >= n_bins) bin = n_bins - 1;
            hist.counts[bin]++;
        }
    }

    return hist;
}

void histogram_destroy(Histogram* hist) {
    if (hist) {
        free(hist->bin_edges);
        free(hist->counts);
    }
}

// ============================================================================
// Polynomial and Interpolation
// ============================================================================

Polynomial polynomial_fit(const double* x, const double* y, size_t n, size_t degree) {
    Polynomial poly;
    poly.degree = degree;
    poly.coefficients = (double*)calloc(degree + 1, sizeof(double));

    // Simplified: would use least squares in practice
    poly.coefficients[0] = compute_mean(y, n);

    return poly;
}

double polynomial_eval(const Polynomial* poly, double x) {
    double result = 0.0;
    double x_power = 1.0;
    for (size_t i = 0; i <= poly->degree; i++) {
        result += poly->coefficients[i] * x_power;
        x_power *= x;
    }
    return result;
}

void polynomial_destroy(Polynomial* poly) {
    if (poly) {
        free(poly->coefficients);
    }
}

SplineInterpolation create_cubic_spline(const double* x, const double* y, size_t n) {
    SplineInterpolation spline;
    spline.n_points = n;
    spline.n_coeffs = 4 * (n - 1);
    spline.x_points = (double*)malloc(n * sizeof(double));
    spline.y_points = (double*)malloc(n * sizeof(double));
    spline.coefficients = (double*)calloc(spline.n_coeffs, sizeof(double));

    memcpy(spline.x_points, x, n * sizeof(double));
    memcpy(spline.y_points, y, n * sizeof(double));

    // Simplified: linear interpolation
    for (size_t i = 0; i < n - 1; i++) {
        spline.coefficients[4*i + 1] = (y[i+1] - y[i]) / (x[i+1] - x[i]);
        spline.coefficients[4*i] = y[i];
    }

    return spline;
}

double spline_eval(const SplineInterpolation* spline, double x) {
    // Find interval
    for (size_t i = 0; i < spline->n_points - 1; i++) {
        if (x >= spline->x_points[i] && x <= spline->x_points[i+1]) {
            double dx = x - spline->x_points[i];
            return spline->coefficients[4*i] + spline->coefficients[4*i+1] * dx;
        }
    }
    return spline->y_points[spline->n_points - 1];
}

void spline_destroy(SplineInterpolation* spline) {
    if (spline) {
        free(spline->x_points);
        free(spline->y_points);
        free(spline->coefficients);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void set_random_seed(uint64_t seed) {
    rng.seed(seed);
}

void fill_random_uniform(double* data, size_t n, double min_val, double max_val) {
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

void fill_random_normal(double* data, size_t n, double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

const char* status_to_string(Status status) {
    switch (status) {
        case Status::SUCCESS: return "SUCCESS";
        case Status::ERROR_INVALID_INPUT: return "ERROR_INVALID_INPUT";
        case Status::ERROR_SINGULAR_MATRIX: return "ERROR_SINGULAR_MATRIX";
        case Status::ERROR_NOT_CONVERGED: return "ERROR_NOT_CONVERGED";
        case Status::ERROR_OUT_OF_MEMORY: return "ERROR_OUT_OF_MEMORY";
        case Status::ERROR_DIMENSION_MISMATCH: return "ERROR_DIMENSION_MISMATCH";
        default: return "UNKNOWN_ERROR";
    }
}

void print_matrix(const DenseMatrix* mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            printf("%10.4f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

void print_vector(const double* vec, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) {
        printf("%.4f", vec[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

} // extern "C"
