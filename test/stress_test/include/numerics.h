#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>

// ============================================================================
// STRESS TEST: Comprehensive C++ Features for Julia FFI
// ============================================================================
// This header tests RepliBuild's ability to handle:
// - Complex struct hierarchies with composition
// - Function pointers (callbacks)
// - Const correctness
// - Arrays and pointers
// - Return value optimization (RVO) for large structs
// - Enums and error codes
// - Template instantiations (simulated via typedefs)
// ============================================================================

// ============================================================================
// ERROR HANDLING
// ============================================================================

enum class Status : int32_t {
    SUCCESS = 0,
    ERROR_INVALID_INPUT = -1,
    ERROR_SINGULAR_MATRIX = -2,
    ERROR_NOT_CONVERGED = -3,
    ERROR_OUT_OF_MEMORY = -4,
    ERROR_DIMENSION_MISMATCH = -5
};

enum OptimizationAlgorithm {
    GRADIENT_DESCENT = 0,
    CONJUGATE_GRADIENT = 1,
    LBFGS = 2,
    NEWTON = 3
};

// ============================================================================
// BASIC TYPES
// ============================================================================

struct Vector3 {
    double x, y, z;
};

struct Matrix3x3 {
    double data[9];  // Column-major order
};

// Large struct to test RVO
struct DenseMatrix {
    double* data;
    size_t rows;
    size_t cols;
    bool owns_data;
};

struct SparseMatrix {
    double* values;
    int32_t* row_indices;
    int32_t* col_pointers;
    size_t nnz;
    size_t rows;
    size_t cols;
};

// ============================================================================
// DECOMPOSITIONS (Tests RVO with complex structs)
// ============================================================================

struct LUDecomposition {
    DenseMatrix L;
    DenseMatrix U;
    int32_t* permutation;
    size_t size;
    Status status;
};

struct QRDecomposition {
    DenseMatrix Q;
    DenseMatrix R;
    size_t m;
    size_t n;
    Status status;
};

struct EigenDecomposition {
    double* eigenvalues;      // Real parts
    double* eigenvalues_imag; // Imaginary parts
    DenseMatrix eigenvectors;
    size_t n;
    Status status;
};

// ============================================================================
// OPTIMIZATION TYPES
// ============================================================================

struct OptimizationState {
    double* x;              // Current point
    double* gradient;       // Gradient at x
    double f_value;         // Function value
    double gradient_norm;
    int32_t iteration;
    int32_t n_evals;
    Status status;
    size_t dimension;
};

struct OptimizationOptions {
    double tolerance;
    double step_size;
    int32_t max_iterations;
    int32_t max_function_evals;
    OptimizationAlgorithm algorithm;
    bool verbose;
};

// ============================================================================
// FUNCTION POINTERS / CALLBACKS
// ============================================================================

// Objective function: f(x, n, user_data) -> double
typedef double (*ObjectiveFunction)(const double* x, size_t n, void* user_data);

// Gradient function: grad_f(x, gradient_out, n, user_data)
typedef void (*GradientFunction)(const double* x, double* gradient_out, size_t n, void* user_data);

// Iteration callback: called after each optimization iteration
// Returns: true to continue, false to stop
typedef bool (*IterationCallback)(const OptimizationState* state, void* user_data);

// Matrix-vector product: y = A*x (for iterative solvers)
typedef void (*MatVecProduct)(const double* x, double* y, size_t n, void* user_data);

// ODE right-hand side: dy/dt = f(t, y, n, user_data)
typedef void (*ODEFunction)(double t, const double* y, double* dydt, size_t n, void* user_data);

// Event detection for ODE integration
typedef double (*EventFunction)(double t, const double* y, size_t n, void* user_data);

// ============================================================================
// C API FUNCTIONS
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// Memory Management
// ----------------------------------------------------------------------------

DenseMatrix dense_matrix_create(size_t rows, size_t cols);
void dense_matrix_destroy(DenseMatrix* mat);
DenseMatrix dense_matrix_copy(const DenseMatrix* src);
void dense_matrix_set_zero(DenseMatrix* mat);
void dense_matrix_set_identity(DenseMatrix* mat);
Status dense_matrix_resize(DenseMatrix* mat, size_t new_rows, size_t new_cols);

SparseMatrix sparse_matrix_create(size_t rows, size_t cols, size_t nnz);
void sparse_matrix_destroy(SparseMatrix* mat);

// ----------------------------------------------------------------------------
// Basic Linear Algebra
// ----------------------------------------------------------------------------

// BLAS Level 1: Vector operations
double vector_dot(const double* x, const double* y, size_t n);
double vector_norm(const double* x, size_t n);
void vector_scale(double* x, double alpha, size_t n);
void vector_axpy(double* y, double alpha, const double* x, size_t n); // y = alpha*x + y
void vector_copy(double* dest, const double* src, size_t n);

// BLAS Level 2: Matrix-vector operations
void matrix_vector_mult(const DenseMatrix* A, const double* x, double* y); // y = A*x
void matrix_vector_mult_add(const DenseMatrix* A, const double* x, double* y, double alpha, double beta); // y = alpha*A*x + beta*y

// BLAS Level 3: Matrix-matrix operations
DenseMatrix matrix_multiply(const DenseMatrix* A, const DenseMatrix* B);
DenseMatrix matrix_add(const DenseMatrix* A, const DenseMatrix* B);
DenseMatrix matrix_transpose(const DenseMatrix* A);
double matrix_trace(const DenseMatrix* A);
double matrix_determinant(const DenseMatrix* A);

// ----------------------------------------------------------------------------
// Linear Algebra Decompositions (Tests RVO!)
// ----------------------------------------------------------------------------

LUDecomposition compute_lu(const DenseMatrix* A);
QRDecomposition compute_qr(const DenseMatrix* A);
EigenDecomposition compute_eigen(const DenseMatrix* A);

Status solve_linear_system_lu(const LUDecomposition* lu, const double* b, double* x, size_t n);
Status solve_linear_system_qr(const QRDecomposition* qr, const double* b, double* x);
Status solve_least_squares(const DenseMatrix* A, const double* b, double* x);

// Iterative solver with callback for matrix-vector products
Status solve_conjugate_gradient(
    MatVecProduct matvec,
    const double* b,
    double* x,
    size_t n,
    double tolerance,
    int32_t max_iterations,
    void* user_data
);

// ----------------------------------------------------------------------------
// Nonlinear Optimization (Heavy callback usage!)
// ----------------------------------------------------------------------------

Status optimize_minimize(
    ObjectiveFunction objective,
    GradientFunction gradient,
    double* x,
    size_t n,
    const OptimizationOptions* options,
    OptimizationState* final_state,
    IterationCallback callback,
    void* user_data
);

Status optimize_minimize_numerical_gradient(
    ObjectiveFunction objective,
    double* x,
    size_t n,
    const OptimizationOptions* options,
    OptimizationState* final_state,
    IterationCallback callback,
    void* user_data
);

// Line search
double line_search_backtracking(
    ObjectiveFunction objective,
    const double* x,
    const double* direction,
    double* x_new,
    size_t n,
    double initial_step,
    void* user_data
);

// ----------------------------------------------------------------------------
// Numerical Integration (ODE Solvers)
// ----------------------------------------------------------------------------

struct ODEResult {
    double* y;              // Solution at t_final
    double* t_values;       // Time points
    double** y_values;      // Solution at each time point
    size_t n_steps;
    size_t dimension;
    Status status;
};

// Solve ODE: dy/dt = f(t, y), y(t0) = y0
ODEResult solve_ode_rk4(
    ODEFunction ode_func,
    double t0,
    double t_final,
    const double* y0,
    size_t n,
    double dt,
    void* user_data
);

// Adaptive timestep solver
ODEResult solve_ode_adaptive(
    ODEFunction ode_func,
    double t0,
    double t_final,
    const double* y0,
    size_t n,
    double tolerance,
    EventFunction event_func,
    void* user_data
);

void ode_result_destroy(ODEResult* result);

// ----------------------------------------------------------------------------
// Signal Processing
// ----------------------------------------------------------------------------

struct FFTResult {
    double* real;
    double* imag;
    size_t n;
};

FFTResult compute_fft(const double* signal, size_t n);
void compute_ifft(const FFTResult* fft_data, double* signal_out);
void fft_result_destroy(FFTResult* result);

void convolve(const double* signal1, size_t n1, const double* signal2, size_t n2, double* result);
void correlate(const double* signal1, const double* signal2, size_t n, double* result);

// ----------------------------------------------------------------------------
// Statistical Functions
// ----------------------------------------------------------------------------

double compute_mean(const double* data, size_t n);
double compute_variance(const double* data, size_t n);
double compute_stddev(const double* data, size_t n);
double compute_median(double* data, size_t n); // Note: modifies data (sorts it)
void compute_quantiles(double* data, size_t n, const double* probabilities, double* quantiles, size_t n_quantiles);

// Histogram
struct Histogram {
    double* bin_edges;
    int32_t* counts;
    size_t n_bins;
};

Histogram compute_histogram(const double* data, size_t n, size_t n_bins, double min_val, double max_val);
void histogram_destroy(Histogram* hist);

// ----------------------------------------------------------------------------
// Polynomial and Interpolation
// ----------------------------------------------------------------------------

struct Polynomial {
    double* coefficients;  // coefficients[i] is coefficient of x^i
    size_t degree;
};

Polynomial polynomial_fit(const double* x, const double* y, size_t n, size_t degree);
double polynomial_eval(const Polynomial* poly, double x);
void polynomial_destroy(Polynomial* poly);

struct SplineInterpolation {
    double* x_points;
    double* y_points;
    double* coefficients;
    size_t n_points;
    size_t n_coeffs;
};

SplineInterpolation create_cubic_spline(const double* x, const double* y, size_t n);
double spline_eval(const SplineInterpolation* spline, double x);
void spline_destroy(SplineInterpolation* spline);

// ----------------------------------------------------------------------------
// Utility Functions
// ----------------------------------------------------------------------------

void set_random_seed(uint64_t seed);
void fill_random_uniform(double* data, size_t n, double min_val, double max_val);
void fill_random_normal(double* data, size_t n, double mean, double stddev);

const char* status_to_string(Status status);
void print_matrix(const DenseMatrix* mat);
void print_vector(const double* vec, size_t n);

} // extern "C"
