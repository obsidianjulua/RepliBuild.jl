# Auto-generated Julia wrapper for stress_test
# Generated: 2026-01-01 16:42:35
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libstress_test.so
# Metadata: compilation_metadata.json
#
# Type Safety: Excellent (~95%) - Types extracted from DWARF debug info
# Ground truth: Types come from compiled binary, not headers
# Manual edits: Minimal to none required

module StressTest

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/stress_test/julia/libstress_test.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.6",
    "clang_version" => "clang version 21.1.6",
    "optimization" => "0",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 57,
    "generated_at" => "2026-01-01T16:42:35.634"
)

# =============================================================================
# Enum Definitions (from DWARF debug info)
# =============================================================================

# C++ enum: OptimizationAlgorithm (underlying type: unsigned int)
@enum OptimizationAlgorithm::Cuint begin
    GRADIENT_DESCENT = 0
    CONJUGATE_GRADIENT = 1
    LBFGS = 2
    NEWTON = 3
end

# C++ enum: Status (underlying type: int32_t)
@enum Status::Int32 begin
    SUCCESS = 0
    ERROR_INVALID_INPUT = -1
    ERROR_SINGULAR_MATRIX = -2
    ERROR_NOT_CONVERGED = -3
    ERROR_OUT_OF_MEMORY = -4
    ERROR_DIMENSION_MISMATCH = -5
end


# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: DenseMatrix (4 members)
mutable struct DenseMatrix
    data::Ptr{Cdouble}
    rows::Csize_t
    cols::Csize_t
    owns_data::Bool
end

# C++ struct: FFTResult (3 members)
mutable struct FFTResult
    real::Ptr{Cdouble}
    imag::Ptr{Cdouble}
    n::Csize_t
end

# C++ struct: Histogram (3 members)
mutable struct Histogram
    bin_edges::Ptr{Cdouble}
    counts::Ptr{Cvoid}
    n_bins::Csize_t
end

# C++ struct: ODEResult (6 members)
mutable struct ODEResult
    y::Ptr{Cdouble}
    t_values::Ptr{Cdouble}
    y_values::Ptr{Ptr{Cdouble}}
    n_steps::Csize_t
    dimension::Csize_t
    status::Status
end

# C++ struct: OptimizationOptions (6 members)
mutable struct OptimizationOptions
    tolerance::Cdouble
    step_size::Cdouble
    max_iterations::Any
    max_function_evals::Any
    algorithm::OptimizationAlgorithm
    verbose::Bool
end

# C++ struct: OptimizationState (8 members)
mutable struct OptimizationState
    x::Ptr{Cdouble}
    gradient::Ptr{Cdouble}
    f_value::Cdouble
    gradient_norm::Cdouble
    iteration::Any
    n_evals::Any
    status::Status
    dimension::Csize_t
end

# C++ struct: Polynomial (2 members)
mutable struct Polynomial
    coefficients::Ptr{Cdouble}
    degree::Csize_t
end

# C++ struct: SparseMatrix (6 members)
mutable struct SparseMatrix
    values::Ptr{Cdouble}
    row_indices::Ptr{Cvoid}
    col_pointers::Ptr{Cvoid}
    nnz::Csize_t
    rows::Csize_t
    cols::Csize_t
end

# C++ struct: SplineInterpolation (5 members)
mutable struct SplineInterpolation
    x_points::Ptr{Cdouble}
    y_points::Ptr{Cdouble}
    coefficients::Ptr{Cdouble}
    n_points::Csize_t
    n_coeffs::Csize_t
end

# C++ struct: __va_list_tag (4 members)
mutable struct __va_list_tag
    gp_offset::Cuint
    fp_offset::Cuint
    overflow_arg_area::Ptr{Cvoid}
    reg_save_area::Ptr{Cvoid}
end

# C++ struct: mersenne_twister_engine<unsigned long, 64UL, 312UL, 156UL, 31UL, 13043109905998158313UL, 29UL, 6148914691236517205UL, 17UL, 8202884508482404352UL, 37UL, 18444473444759240704UL, 43UL, 6364136223846793005UL> (2 members)
mutable struct mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL
    _M_x::NTuple{312, Culong}
    _M_p::Csize_t
end

# C++ struct: mersenne_twister_engine<unsigned long, 64UL, 312UL, 156UL, 31UL, 13043109905998158313UL, 29UL, 6148914691236517205UL, 17UL, 8202884508482404352UL, 37UL, 18444473444759240704UL, 43UL, 6364136223846793005UL>, double> (1 members)
mutable struct mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL_double
    _M_g::Ref{mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL}
end

# C++ struct: param_type (4 members)
mutable struct param_type
    _M_mean::Cdouble
    _M_stddev::Cdouble
    _M_saved::Cdouble
    _M_saved_available::Bool
end

# C++ struct: EigenDecomposition (5 members)
mutable struct EigenDecomposition
    eigenvalues::Ptr{Cdouble}
    eigenvalues_imag::Ptr{Cdouble}
    eigenvectors::DenseMatrix
    n::Csize_t
    status::Status
end

# C++ struct: LUDecomposition (5 members)
mutable struct LUDecomposition
    L::DenseMatrix
    U::DenseMatrix
    permutation::Ptr{Cvoid}
    size::Csize_t
    status::Status
end

# C++ struct: QRDecomposition (5 members)
mutable struct QRDecomposition
    Q::DenseMatrix
    R::DenseMatrix
    m::Csize_t
    n::Csize_t
    status::Status
end

# C++ struct: normal_distribution<double> (1 members)
mutable struct normal_distribution_double
    _M_param::param_type
end

# C++ struct: uniform_real_distribution<double> (1 members)
mutable struct uniform_real_distribution_double
    _M_param::param_type
end


export compute_eigen, compute_fft, compute_histogram, compute_ifft, compute_lu, compute_mean, compute_median, compute_qr, compute_quantiles, compute_stddev, compute_variance, convolve, correlate, create_cubic_spline, dense_matrix_copy, dense_matrix_create, dense_matrix_destroy, dense_matrix_resize, dense_matrix_set_identity, dense_matrix_set_zero, fft_result_destroy, fill_random_normal, fill_random_uniform, histogram_destroy, line_search_backtracking, matrix_add, matrix_determinant, matrix_multiply, matrix_trace, matrix_transpose, matrix_vector_mult, matrix_vector_mult_add, ode_result_destroy, optimize_minimize, optimize_minimize_numerical_gradient, polynomial_destroy, polynomial_eval, polynomial_fit, print_matrix, print_vector, set_random_seed, solve_conjugate_gradient, solve_least_squares, solve_linear_system_lu, solve_linear_system_qr, solve_ode_adaptive, solve_ode_rk4, sparse_matrix_create, sparse_matrix_destroy, spline_destroy, spline_eval, status_to_string, vector_axpy, vector_copy, vector_dot, vector_norm, vector_scale, OptimizationAlgorithm, GRADIENT_DESCENT, CONJUGATE_GRADIENT, LBFGS, NEWTON, Status, SUCCESS, ERROR_INVALID_INPUT, ERROR_SINGULAR_MATRIX, ERROR_NOT_CONVERGED, ERROR_OUT_OF_MEMORY, ERROR_DIMENSION_MISMATCH, LUDecomposition, ODEResult, EigenDecomposition, DenseMatrix, SparseMatrix, mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL, QRDecomposition, uniform_real_distribution_double, FFTResult, mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL_double, __va_list_tag, OptimizationOptions, Histogram, normal_distribution_double, Polynomial, OptimizationState, param_type, SplineInterpolation

"""
    compute_eigen(A::Ptr{DenseMatrix}) -> EigenDecomposition

Wrapper for C++ function: `compute_eigen`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `EigenDecomposition`

# Metadata
- Mangled symbol: `compute_eigen`
- Type safety:  From compilation
"""

function compute_eigen(A::Ptr{DenseMatrix})::EigenDecomposition
    ccall((:compute_eigen, LIBRARY_PATH), EigenDecomposition, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function compute_eigen(A::DenseMatrix)::EigenDecomposition
    return ccall((:compute_eigen, LIBRARY_PATH), EigenDecomposition, (Ptr{DenseMatrix},), Ref(A))
end

"""
    compute_fft(signal::Ptr{Cdouble}, n::Csize_t) -> FFTResult

Wrapper for C++ function: `compute_fft`

# Arguments
- `signal::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `FFTResult`

# Metadata
- Mangled symbol: `compute_fft`
- Type safety:  From compilation
"""

function compute_fft(signal::Ptr{Cdouble}, n::Csize_t)::FFTResult
    ccall((:compute_fft, LIBRARY_PATH), FFTResult, (Ptr{Cdouble}, Csize_t,), signal, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_fft(signal::Vector{Float64}, n::Csize_t)::FFTResult
    return GC.@preserve signal begin
        ccall((:compute_fft, LIBRARY_PATH), FFTResult, (Ptr{Cdouble}, Csize_t,), pointer(signal), n)
    end
end

"""
    compute_histogram(data::Ptr{Cdouble}, n::Csize_t, n_bins::Csize_t, min_val::Cdouble, max_val::Cdouble) -> Histogram

Wrapper for C++ function: `compute_histogram`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`
- `n_bins::Csize_t`
- `min_val::Cdouble`
- `max_val::Cdouble`

# Returns
- `Histogram`

# Metadata
- Mangled symbol: `compute_histogram`
- Type safety:  From compilation
"""

function compute_histogram(data::Ptr{Cdouble}, n::Csize_t, n_bins::Csize_t, min_val::Cdouble, max_val::Cdouble)::Histogram
    ccall((:compute_histogram, LIBRARY_PATH), Histogram, (Ptr{Cdouble}, Csize_t, Csize_t, Cdouble, Cdouble,), data, n, n_bins, min_val, max_val)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_histogram(data::Vector{Float64}, n::Csize_t, n_bins::Csize_t, min_val::Cdouble, max_val::Cdouble)::Histogram
    return GC.@preserve data begin
        ccall((:compute_histogram, LIBRARY_PATH), Histogram, (Ptr{Cdouble}, Csize_t, Csize_t, Cdouble, Cdouble,), pointer(data), n, n_bins, min_val, max_val)
    end
end

"""
    compute_ifft(fft_data::Ptr{FFTResult}, signal_out::Ptr{Cdouble}) -> Cvoid

Wrapper for C++ function: `compute_ifft`

# Arguments
- `fft_data::Ptr{FFTResult}`
- `signal_out::Ptr{Cdouble}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `compute_ifft`
- Type safety:  From compilation
"""

function compute_ifft(fft_data::Ptr{FFTResult}, signal_out::Ptr{Cdouble})::Cvoid
    ccall((:compute_ifft, LIBRARY_PATH), Cvoid, (Ptr{FFTResult}, Ptr{Cdouble},), fft_data, signal_out)
end

# Convenience wrapper - accepts structs directly instead of pointers
function compute_ifft(fft_data::FFTResult, signal_out::Ptr{Cdouble})::Cvoid
    return ccall((:compute_ifft, LIBRARY_PATH), Cvoid, (Ptr{FFTResult}, Ptr{Cdouble},), Ref(fft_data), signal_out)
end

"""
    compute_lu(A::Ptr{DenseMatrix}) -> LUDecomposition

Wrapper for C++ function: `compute_lu`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `LUDecomposition`

# Metadata
- Mangled symbol: `compute_lu`
- Type safety:  From compilation
"""

function compute_lu(A::Ptr{DenseMatrix})::LUDecomposition
    ccall((:compute_lu, LIBRARY_PATH), LUDecomposition, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function compute_lu(A::DenseMatrix)::LUDecomposition
    return ccall((:compute_lu, LIBRARY_PATH), LUDecomposition, (Ptr{DenseMatrix},), Ref(A))
end

"""
    compute_mean(data::Ptr{Cdouble}, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_mean`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_mean`
- Type safety:  From compilation
"""

function compute_mean(data::Ptr{Cdouble}, n::Csize_t)::Cdouble
    ccall((:compute_mean, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_mean(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_mean, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    compute_median(data::Ptr{Cdouble}, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_median`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_median`
- Type safety:  From compilation
"""

function compute_median(data::Ptr{Cdouble}, n::Csize_t)::Cdouble
    ccall((:compute_median, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_median(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_median, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    compute_qr(A::Ptr{DenseMatrix}) -> QRDecomposition

Wrapper for C++ function: `compute_qr`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `QRDecomposition`

# Metadata
- Mangled symbol: `compute_qr`
- Type safety:  From compilation
"""

function compute_qr(A::Ptr{DenseMatrix})::QRDecomposition
    ccall((:compute_qr, LIBRARY_PATH), QRDecomposition, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function compute_qr(A::DenseMatrix)::QRDecomposition
    return ccall((:compute_qr, LIBRARY_PATH), QRDecomposition, (Ptr{DenseMatrix},), Ref(A))
end

"""
    compute_quantiles(data::Ptr{Cdouble}, n::Csize_t, probabilities::Ptr{Cdouble}, quantiles::Ptr{Cdouble}, n_quantiles::Csize_t) -> Cvoid

Wrapper for C++ function: `compute_quantiles`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`
- `probabilities::Ptr{Cdouble}`
- `quantiles::Ptr{Cdouble}`
- `n_quantiles::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `compute_quantiles`
- Type safety:  From compilation
"""

function compute_quantiles(data::Ptr{Cdouble}, n::Csize_t, probabilities::Ptr{Cdouble}, quantiles::Ptr{Cdouble}, n_quantiles::Csize_t)::Cvoid
    ccall((:compute_quantiles, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), data, n, probabilities, quantiles, n_quantiles)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_quantiles(data::Vector{Float64}, n::Csize_t, probabilities::Ptr{Cdouble}, quantiles::Ptr{Cdouble}, n_quantiles::Csize_t)::Cvoid
    return GC.@preserve data begin
        ccall((:compute_quantiles, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), pointer(data), n, probabilities, quantiles, n_quantiles)
    end
end

"""
    compute_stddev(data::Ptr{Cdouble}, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_stddev`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_stddev`
- Type safety:  From compilation
"""

function compute_stddev(data::Ptr{Cdouble}, n::Csize_t)::Cdouble
    ccall((:compute_stddev, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_stddev(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_stddev, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    compute_variance(data::Ptr{Cdouble}, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_variance`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_variance`
- Type safety:  From compilation
"""

function compute_variance(data::Ptr{Cdouble}, n::Csize_t)::Cdouble
    ccall((:compute_variance, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_variance(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_variance, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    convolve(signal1::Ptr{Cdouble}, n1::Csize_t, signal2::Ptr{Cdouble}, n2::Csize_t, result::Ptr{Cdouble}) -> Cvoid

Wrapper for C++ function: `convolve`

# Arguments
- `signal1::Ptr{Cdouble}`
- `n1::Csize_t`
- `signal2::Ptr{Cdouble}`
- `n2::Csize_t`
- `result::Ptr{Cdouble}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `convolve`
- Type safety:  From compilation
"""

function convolve(signal1::Ptr{Cdouble}, n1::Csize_t, signal2::Ptr{Cdouble}, n2::Csize_t, result::Ptr{Cdouble})::Cvoid
    ccall((:convolve, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), signal1, n1, signal2, n2, result)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function convolve(signal1::Vector{Float64}, n1::Csize_t, signal2::Vector{Float64}, n2::Csize_t, result::Ptr{Cdouble})::Cvoid
    return GC.@preserve signal1 signal2 begin
        ccall((:convolve, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), pointer(signal1), n1, pointer(signal2), n2, result)
    end
end

"""
    correlate(signal1::Ptr{Cdouble}, signal2::Ptr{Cdouble}, n::Csize_t, result::Ptr{Cdouble}) -> Cvoid

Wrapper for C++ function: `correlate`

# Arguments
- `signal1::Ptr{Cdouble}`
- `signal2::Ptr{Cdouble}`
- `n::Csize_t`
- `result::Ptr{Cdouble}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `correlate`
- Type safety:  From compilation
"""

function correlate(signal1::Ptr{Cdouble}, signal2::Ptr{Cdouble}, n::Csize_t, result::Ptr{Cdouble})::Cvoid
    ccall((:correlate, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), signal1, signal2, n, result)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function correlate(signal1::Vector{Float64}, signal2::Vector{Float64}, n::Csize_t, result::Ptr{Cdouble})::Cvoid
    return GC.@preserve signal1 signal2 begin
        ccall((:correlate, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), pointer(signal1), pointer(signal2), n, result)
    end
end

"""
    create_cubic_spline(x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t) -> SplineInterpolation

Wrapper for C++ function: `create_cubic_spline`

# Arguments
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `SplineInterpolation`

# Metadata
- Mangled symbol: `create_cubic_spline`
- Type safety:  From compilation
"""

function create_cubic_spline(x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t)::SplineInterpolation
    ccall((:create_cubic_spline, LIBRARY_PATH), SplineInterpolation, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), x, y, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function create_cubic_spline(x::Vector{Float64}, y::Vector{Float64}, n::Csize_t)::SplineInterpolation
    return GC.@preserve x y begin
        ccall((:create_cubic_spline, LIBRARY_PATH), SplineInterpolation, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), pointer(x), pointer(y), n)
    end
end

"""
    dense_matrix_copy(src::Ptr{DenseMatrix}) -> DenseMatrix

Wrapper for C++ function: `dense_matrix_copy`

# Arguments
- `src::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `dense_matrix_copy`
- Type safety:  From compilation
"""

function dense_matrix_copy(src::Ptr{DenseMatrix})::DenseMatrix
    ccall((:dense_matrix_copy, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix},), src)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_copy(src::DenseMatrix)::DenseMatrix
    return ccall((:dense_matrix_copy, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix},), Ref(src))
end

"""
    dense_matrix_create(rows::Csize_t, cols::Csize_t) -> DenseMatrix

Wrapper for C++ function: `dense_matrix_create`

# Arguments
- `rows::Csize_t`
- `cols::Csize_t`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `dense_matrix_create`
- Type safety:  From compilation
"""

function dense_matrix_create(rows::Csize_t, cols::Csize_t)::DenseMatrix
    ccall((:dense_matrix_create, LIBRARY_PATH), DenseMatrix, (Csize_t, Csize_t,), rows, cols)
end

"""
    dense_matrix_destroy(mat::Ptr{DenseMatrix}) -> Cvoid

Wrapper for C++ function: `dense_matrix_destroy`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `dense_matrix_destroy`
- Type safety:  From compilation
"""

function dense_matrix_destroy(mat::Ptr{DenseMatrix})::Cvoid
    ccall((:dense_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_destroy(mat::DenseMatrix)::Cvoid
    return ccall((:dense_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    dense_matrix_resize(mat::Ptr{DenseMatrix}, new_rows::Csize_t, new_cols::Csize_t) -> Any

Wrapper for C++ function: `dense_matrix_resize`

# Arguments
- `mat::Ptr{DenseMatrix}`
- `new_rows::Csize_t`
- `new_cols::Csize_t`

# Returns
- `Any`

# Metadata
- Mangled symbol: `dense_matrix_resize`
- Type safety:  From compilation
"""

function dense_matrix_resize(mat::Ptr{DenseMatrix}, new_rows::Csize_t, new_cols::Csize_t)::Status
    return ccall((:dense_matrix_resize, LIBRARY_PATH), Status, (Ptr{DenseMatrix}, Csize_t, Csize_t,), mat, new_rows, new_cols)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_resize(mat::DenseMatrix, new_rows::Csize_t, new_cols::Csize_t)::Any
    return ccall((:dense_matrix_resize, LIBRARY_PATH), Any, (Ptr{DenseMatrix}, Csize_t, Csize_t,), Ref(mat), new_rows, new_cols)
end

"""
    dense_matrix_set_identity(mat::Ptr{DenseMatrix}) -> Cvoid

Wrapper for C++ function: `dense_matrix_set_identity`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `dense_matrix_set_identity`
- Type safety:  From compilation
"""

function dense_matrix_set_identity(mat::Ptr{DenseMatrix})::Cvoid
    ccall((:dense_matrix_set_identity, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_set_identity(mat::DenseMatrix)::Cvoid
    return ccall((:dense_matrix_set_identity, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    dense_matrix_set_zero(mat::Ptr{DenseMatrix}) -> Cvoid

Wrapper for C++ function: `dense_matrix_set_zero`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `dense_matrix_set_zero`
- Type safety:  From compilation
"""

function dense_matrix_set_zero(mat::Ptr{DenseMatrix})::Cvoid
    ccall((:dense_matrix_set_zero, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_set_zero(mat::DenseMatrix)::Cvoid
    return ccall((:dense_matrix_set_zero, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    fft_result_destroy(result::Ptr{FFTResult}) -> Cvoid

Wrapper for C++ function: `fft_result_destroy`

# Arguments
- `result::Ptr{FFTResult}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `fft_result_destroy`
- Type safety:  From compilation
"""

function fft_result_destroy(result::Ptr{FFTResult})::Cvoid
    ccall((:fft_result_destroy, LIBRARY_PATH), Cvoid, (Ptr{FFTResult},), result)
end

# Convenience wrapper - accepts structs directly instead of pointers
function fft_result_destroy(result::FFTResult)::Cvoid
    return ccall((:fft_result_destroy, LIBRARY_PATH), Cvoid, (Ptr{FFTResult},), Ref(result))
end

"""
    fill_random_normal(data::Ptr{Cdouble}, n::Csize_t, mean::Cdouble, stddev::Cdouble) -> Cvoid

Wrapper for C++ function: `fill_random_normal`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`
- `mean::Cdouble`
- `stddev::Cdouble`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `fill_random_normal`
- Type safety:  From compilation
"""

function fill_random_normal(data::Ptr{Cdouble}, n::Csize_t, mean::Cdouble, stddev::Cdouble)::Cvoid
    ccall((:fill_random_normal, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), data, n, mean, stddev)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function fill_random_normal(data::Vector{Float64}, n::Csize_t, mean::Cdouble, stddev::Cdouble)::Cvoid
    return GC.@preserve data begin
        ccall((:fill_random_normal, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), pointer(data), n, mean, stddev)
    end
end

"""
    fill_random_uniform(data::Ptr{Cdouble}, n::Csize_t, min_val::Cdouble, max_val::Cdouble) -> Cvoid

Wrapper for C++ function: `fill_random_uniform`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`
- `min_val::Cdouble`
- `max_val::Cdouble`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `fill_random_uniform`
- Type safety:  From compilation
"""

function fill_random_uniform(data::Ptr{Cdouble}, n::Csize_t, min_val::Cdouble, max_val::Cdouble)::Cvoid
    ccall((:fill_random_uniform, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), data, n, min_val, max_val)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function fill_random_uniform(data::Vector{Float64}, n::Csize_t, min_val::Cdouble, max_val::Cdouble)::Cvoid
    return GC.@preserve data begin
        ccall((:fill_random_uniform, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), pointer(data), n, min_val, max_val)
    end
end

"""
    histogram_destroy(hist::Ptr{Histogram}) -> Cvoid

Wrapper for C++ function: `histogram_destroy`

# Arguments
- `hist::Ptr{Histogram}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `histogram_destroy`
- Type safety:  From compilation
"""

function histogram_destroy(hist::Ptr{Histogram})::Cvoid
    ccall((:histogram_destroy, LIBRARY_PATH), Cvoid, (Ptr{Histogram},), hist)
end

# Convenience wrapper - accepts structs directly instead of pointers
function histogram_destroy(hist::Histogram)::Cvoid
    return ccall((:histogram_destroy, LIBRARY_PATH), Cvoid, (Ptr{Histogram},), Ref(hist))
end

"""
    line_search_backtracking(objective::Ptr{Cvoid}, x::Ptr{Cdouble}, direction::Ptr{Cdouble}, x_new::Ptr{Cdouble}, n::Csize_t, initial_step::Cdouble, user_data::Ptr{Cvoid}) -> Cdouble

Wrapper for C++ function: `line_search_backtracking`

# Arguments
- `objective::Ptr{Cvoid}` - Callback function
- `x::Ptr{Cdouble}`
- `direction::Ptr{Cdouble}`
- `x_new::Ptr{Cdouble}`
- `n::Csize_t`
- `initial_step::Cdouble`
- `user_data::Ptr{Cvoid}`

# Returns
- `Cdouble`

            # Callback Signatures
**Callback `objective`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cdouble, (Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `line_search_backtracking`
- Type safety:  From compilation
"""

function line_search_backtracking(objective::Ptr{Cvoid}, x::Ptr{Cdouble}, direction::Ptr{Cdouble}, x_new::Ptr{Cdouble}, n::Csize_t, initial_step::Cdouble, user_data::Ptr{Cvoid})::Cdouble
    ccall((:line_search_backtracking, LIBRARY_PATH), Cdouble, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid},), objective, x, direction, x_new, n, initial_step, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function line_search_backtracking(objective::Ptr{Cvoid}, x::Vector{Float64}, direction::Ptr{Cdouble}, x_new::Vector{Float64}, n::Csize_t, initial_step::Cdouble, user_data::Ptr{Cvoid})::Cdouble
    return GC.@preserve x x_new begin
        ccall((:line_search_backtracking, LIBRARY_PATH), Cdouble, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid},), objective, pointer(x), direction, pointer(x_new), n, initial_step, user_data)
    end
end

"""
    matrix_add(A::Ptr{DenseMatrix}, B::Ptr{DenseMatrix}) -> DenseMatrix

Wrapper for C++ function: `matrix_add`

# Arguments
- `A::Ptr{DenseMatrix}`
- `B::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `matrix_add`
- Type safety:  From compilation
"""

function matrix_add(A::Ptr{DenseMatrix}, B::Ptr{DenseMatrix})::DenseMatrix
    ccall((:matrix_add, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix}, Ptr{DenseMatrix},), A, B)
end

# Convenience wrapper - accepts structs directly instead of pointers
function matrix_add(A::DenseMatrix, B::DenseMatrix)::DenseMatrix
    return ccall((:matrix_add, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix}, Ptr{DenseMatrix},), Ref(A), Ref(B))
end

"""
    matrix_determinant(A::Ptr{DenseMatrix}) -> Cdouble

Wrapper for C++ function: `matrix_determinant`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `matrix_determinant`
- Type safety:  From compilation
"""

function matrix_determinant(A::Ptr{DenseMatrix})::Cdouble
    ccall((:matrix_determinant, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function matrix_determinant(A::DenseMatrix)::Cdouble
    return ccall((:matrix_determinant, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), Ref(A))
end

"""
    matrix_multiply(A::Ptr{DenseMatrix}, B::Ptr{DenseMatrix}) -> DenseMatrix

Wrapper for C++ function: `matrix_multiply`

# Arguments
- `A::Ptr{DenseMatrix}`
- `B::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `matrix_multiply`
- Type safety:  From compilation
"""

function matrix_multiply(A::Ptr{DenseMatrix}, B::Ptr{DenseMatrix})::DenseMatrix
    ccall((:matrix_multiply, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix}, Ptr{DenseMatrix},), A, B)
end

# Convenience wrapper - accepts structs directly instead of pointers
function matrix_multiply(A::DenseMatrix, B::DenseMatrix)::DenseMatrix
    return ccall((:matrix_multiply, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix}, Ptr{DenseMatrix},), Ref(A), Ref(B))
end

"""
    matrix_trace(A::Ptr{DenseMatrix}) -> Cdouble

Wrapper for C++ function: `matrix_trace`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `matrix_trace`
- Type safety:  From compilation
"""

function matrix_trace(A::Ptr{DenseMatrix})::Cdouble
    ccall((:matrix_trace, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function matrix_trace(A::DenseMatrix)::Cdouble
    return ccall((:matrix_trace, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), Ref(A))
end

"""
    matrix_transpose(A::Ptr{DenseMatrix}) -> DenseMatrix

Wrapper for C++ function: `matrix_transpose`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `matrix_transpose`
- Type safety:  From compilation
"""

function matrix_transpose(A::Ptr{DenseMatrix})::DenseMatrix
    ccall((:matrix_transpose, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function matrix_transpose(A::DenseMatrix)::DenseMatrix
    return ccall((:matrix_transpose, LIBRARY_PATH), DenseMatrix, (Ptr{DenseMatrix},), Ref(A))
end

"""
    matrix_vector_mult(A::Ptr{DenseMatrix}, x::Ptr{Cdouble}, y::Ptr{Cdouble}) -> Cvoid

Wrapper for C++ function: `matrix_vector_mult`

# Arguments
- `A::Ptr{DenseMatrix}`
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `matrix_vector_mult`
- Type safety:  From compilation
"""

function matrix_vector_mult(A::Ptr{DenseMatrix}, x::Ptr{Cdouble}, y::Ptr{Cdouble})::Cvoid
    ccall((:matrix_vector_mult, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), A, x, y)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function matrix_vector_mult(A::DenseMatrix, x::Vector{Float64}, y::Vector{Float64})::Cvoid
    return GC.@preserve x y begin
        ccall((:matrix_vector_mult, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), Ref(A), pointer(x), pointer(y))
    end
end

"""
    matrix_vector_mult_add(A::Ptr{DenseMatrix}, x::Ptr{Cdouble}, y::Ptr{Cdouble}, alpha::Cdouble, beta::Cdouble) -> Cvoid

Wrapper for C++ function: `matrix_vector_mult_add`

# Arguments
- `A::Ptr{DenseMatrix}`
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`
- `alpha::Cdouble`
- `beta::Cdouble`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `matrix_vector_mult_add`
- Type safety:  From compilation
"""

function matrix_vector_mult_add(A::Ptr{DenseMatrix}, x::Ptr{Cdouble}, y::Ptr{Cdouble}, alpha::Cdouble, beta::Cdouble)::Cvoid
    ccall((:matrix_vector_mult_add, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,), A, x, y, alpha, beta)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function matrix_vector_mult_add(A::DenseMatrix, x::Vector{Float64}, y::Vector{Float64}, alpha::Cdouble, beta::Cdouble)::Cvoid
    return GC.@preserve x y begin
        ccall((:matrix_vector_mult_add, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,), Ref(A), pointer(x), pointer(y), alpha, beta)
    end
end

"""
    ode_result_destroy(result::Ptr{ODEResult}) -> Cvoid

Wrapper for C++ function: `ode_result_destroy`

# Arguments
- `result::Ptr{ODEResult}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `ode_result_destroy`
- Type safety:  From compilation
"""

function ode_result_destroy(result::Ptr{ODEResult})::Cvoid
    ccall((:ode_result_destroy, LIBRARY_PATH), Cvoid, (Ptr{ODEResult},), result)
end

"""
    optimize_minimize(objective::Ptr{Cvoid}, gradient::Ptr{Cvoid}, x::Ptr{Cdouble}, n::Csize_t, options::Ptr{OptimizationOptions}, final_state::Ptr{OptimizationState}, callback::Ptr{Cvoid}, user_data::Ptr{Cvoid}) -> Any

Wrapper for C++ function: `optimize_minimize`

# Arguments
- `objective::Ptr{Cvoid}` - Callback function
- `gradient::Ptr{Cvoid}` - Callback function
- `x::Ptr{Cdouble}`
- `n::Csize_t`
- `options::Ptr{OptimizationOptions}`
- `final_state::Ptr{OptimizationState}`
- `callback::Ptr{Cvoid}` - Callback function
- `user_data::Ptr{Cvoid}`

# Returns
- `Any`

            # Callback Signatures
**Callback `objective`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cdouble, (Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

**Callback `gradient`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

**Callback `callback`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Bool, (Ptr{OptimizationState}, Ptr{Cvoid},)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `optimize_minimize`
- Type safety:  From compilation
"""

function optimize_minimize(objective::Ptr{Cvoid}, gradient::Ptr{Cvoid}, x::Ptr{Cdouble}, n::Csize_t, options::Ptr{OptimizationOptions}, final_state::Ptr{OptimizationState}, callback::Ptr{Cvoid}, user_data::Ptr{Cvoid})::Status
    return ccall((:optimize_minimize, LIBRARY_PATH), Status, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{OptimizationOptions}, Ptr{OptimizationState}, Ptr{Cvoid}, Ptr{Cvoid},), objective, gradient, x, n, options, final_state, callback, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function optimize_minimize(objective::Ptr{Cvoid}, gradient::Ptr{Cvoid}, x::Vector{Float64}, n::Csize_t, options::OptimizationOptions, final_state::OptimizationState, callback::Ptr{Cvoid}, user_data::Ptr{Cvoid})::Any
    return GC.@preserve x begin
        ccall((:optimize_minimize, LIBRARY_PATH), Any, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{OptimizationOptions}, Ptr{OptimizationState}, Ptr{Cvoid}, Ptr{Cvoid},), objective, gradient, pointer(x), n, Ref(options), Ref(final_state), callback, user_data)
    end
end

"""
    optimize_minimize_numerical_gradient(objective::Ptr{Cvoid}, x::Ptr{Cdouble}, n::Csize_t, options::Ptr{OptimizationOptions}, final_state::Ptr{OptimizationState}, callback::Ptr{Cvoid}, user_data::Ptr{Cvoid}) -> Any

Wrapper for C++ function: `optimize_minimize_numerical_gradient`

# Arguments
- `objective::Ptr{Cvoid}` - Callback function
- `x::Ptr{Cdouble}`
- `n::Csize_t`
- `options::Ptr{OptimizationOptions}`
- `final_state::Ptr{OptimizationState}`
- `callback::Ptr{Cvoid}` - Callback function
- `user_data::Ptr{Cvoid}`

# Returns
- `Any`

            # Callback Signatures
**Callback `objective`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cdouble, (Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

**Callback `callback`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Bool, (Ptr{OptimizationState}, Ptr{Cvoid},)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `optimize_minimize_numerical_gradient`
- Type safety:  From compilation
"""

function optimize_minimize_numerical_gradient(objective::Ptr{Cvoid}, x::Ptr{Cdouble}, n::Csize_t, options::Ptr{OptimizationOptions}, final_state::Ptr{OptimizationState}, callback::Ptr{Cvoid}, user_data::Ptr{Cvoid})::Status
    return ccall((:optimize_minimize_numerical_gradient, LIBRARY_PATH), Status, (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{OptimizationOptions}, Ptr{OptimizationState}, Ptr{Cvoid}, Ptr{Cvoid},), objective, x, n, options, final_state, callback, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function optimize_minimize_numerical_gradient(objective::Ptr{Cvoid}, x::Vector{Float64}, n::Csize_t, options::OptimizationOptions, final_state::OptimizationState, callback::Ptr{Cvoid}, user_data::Ptr{Cvoid})::Any
    return GC.@preserve x begin
        ccall((:optimize_minimize_numerical_gradient, LIBRARY_PATH), Any, (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{OptimizationOptions}, Ptr{OptimizationState}, Ptr{Cvoid}, Ptr{Cvoid},), objective, pointer(x), n, Ref(options), Ref(final_state), callback, user_data)
    end
end

"""
    polynomial_destroy(poly::Ptr{Polynomial}) -> Cvoid

Wrapper for C++ function: `polynomial_destroy`

# Arguments
- `poly::Ptr{Polynomial}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `polynomial_destroy`
- Type safety:  From compilation
"""

function polynomial_destroy(poly::Ptr{Polynomial})::Cvoid
    ccall((:polynomial_destroy, LIBRARY_PATH), Cvoid, (Ptr{Polynomial},), poly)
end

"""
    polynomial_eval(poly::Ptr{Polynomial}, x::Cdouble) -> Cdouble

Wrapper for C++ function: `polynomial_eval`

# Arguments
- `poly::Ptr{Polynomial}`
- `x::Cdouble`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `polynomial_eval`
- Type safety:  From compilation
"""

function polynomial_eval(poly::Ptr{Polynomial}, x::Cdouble)::Cdouble
    ccall((:polynomial_eval, LIBRARY_PATH), Cdouble, (Ptr{Polynomial}, Cdouble,), poly, x)
end

"""
    polynomial_fit(x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t, degree::Csize_t) -> Polynomial

Wrapper for C++ function: `polynomial_fit`

# Arguments
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`
- `n::Csize_t`
- `degree::Csize_t`

# Returns
- `Polynomial`

# Metadata
- Mangled symbol: `polynomial_fit`
- Type safety:  From compilation
"""

function polynomial_fit(x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t, degree::Csize_t)::Polynomial
    ccall((:polynomial_fit, LIBRARY_PATH), Polynomial, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Csize_t,), x, y, n, degree)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function polynomial_fit(x::Vector{Float64}, y::Vector{Float64}, n::Csize_t, degree::Csize_t)::Polynomial
    return GC.@preserve x y begin
        ccall((:polynomial_fit, LIBRARY_PATH), Polynomial, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Csize_t,), pointer(x), pointer(y), n, degree)
    end
end

"""
    print_matrix(mat::Ptr{DenseMatrix}) -> Cvoid

Wrapper for C++ function: `print_matrix`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `print_matrix`
- Type safety:  From compilation
"""

function print_matrix(mat::Ptr{DenseMatrix})::Cvoid
    ccall((:print_matrix, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function print_matrix(mat::DenseMatrix)::Cvoid
    return ccall((:print_matrix, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    print_vector(vec::Ptr{Cdouble}, n::Csize_t) -> Cvoid

Wrapper for C++ function: `print_vector`

# Arguments
- `vec::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `print_vector`
- Type safety:  From compilation
"""

function print_vector(vec::Ptr{Cdouble}, n::Csize_t)::Cvoid
    ccall((:print_vector, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t,), vec, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function print_vector(vec::Vector{Float64}, n::Csize_t)::Cvoid
    return GC.@preserve vec begin
        ccall((:print_vector, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t,), pointer(vec), n)
    end
end

"""
    set_random_seed(seed::Any) -> Cvoid

Wrapper for C++ function: `set_random_seed`

# Arguments
- `seed::Any`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `set_random_seed`
- Type safety:  From compilation
"""

function set_random_seed(seed::Any)::Cvoid
    ccall((:set_random_seed, LIBRARY_PATH), Cvoid, (Any,), seed)
end

"""
    solve_conjugate_gradient(matvec::Ptr{Cvoid}, b::Ptr{Cdouble}, x::Ptr{Cdouble}, n::Csize_t, tolerance::Cdouble, max_iterations::Any, user_data::Ptr{Cvoid}) -> Any

Wrapper for C++ function: `solve_conjugate_gradient`

# Arguments
- `matvec::Ptr{Cvoid}` - Callback function
- `b::Ptr{Cdouble}`
- `x::Ptr{Cdouble}`
- `n::Csize_t`
- `tolerance::Cdouble`
- `max_iterations::Any`
- `user_data::Ptr{Cvoid}`

# Returns
- `Any`

            # Callback Signatures
**Callback `matvec`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `solve_conjugate_gradient`
- Type safety:  From compilation
"""

function solve_conjugate_gradient(matvec::Ptr{Cvoid}, b::Ptr{Cdouble}, x::Ptr{Cdouble}, n::Csize_t, tolerance::Cdouble, max_iterations::Any, user_data::Ptr{Cvoid})::Status
    return ccall((:solve_conjugate_gradient, LIBRARY_PATH), Status, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Any, Ptr{Cvoid},), matvec, b, x, n, tolerance, max_iterations, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_conjugate_gradient(matvec::Ptr{Cvoid}, b::Ptr{Cdouble}, x::Vector{Float64}, n::Csize_t, tolerance::Cdouble, max_iterations::Any, user_data::Ptr{Cvoid})::Any
    return GC.@preserve x begin
        ccall((:solve_conjugate_gradient, LIBRARY_PATH), Any, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Any, Ptr{Cvoid},), matvec, b, pointer(x), n, tolerance, max_iterations, user_data)
    end
end

"""
    solve_least_squares(A::Ptr{DenseMatrix}, b::Ptr{Cdouble}, x::Ptr{Cdouble}) -> Any

Wrapper for C++ function: `solve_least_squares`

# Arguments
- `A::Ptr{DenseMatrix}`
- `b::Ptr{Cdouble}`
- `x::Ptr{Cdouble}`

# Returns
- `Any`

# Metadata
- Mangled symbol: `solve_least_squares`
- Type safety:  From compilation
"""

function solve_least_squares(A::Ptr{DenseMatrix}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Status
    return ccall((:solve_least_squares, LIBRARY_PATH), Status, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), A, b, x)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_least_squares(A::DenseMatrix, b::Ptr{Cdouble}, x::Vector{Float64})::Any
    return GC.@preserve x begin
        ccall((:solve_least_squares, LIBRARY_PATH), Any, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), Ref(A), b, pointer(x))
    end
end

"""
    solve_linear_system_lu(lu::Ptr{LUDecomposition}, b::Ptr{Cdouble}, x::Ptr{Cdouble}, n::Csize_t) -> Any

Wrapper for C++ function: `solve_linear_system_lu`

# Arguments
- `lu::Ptr{LUDecomposition}`
- `b::Ptr{Cdouble}`
- `x::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Any`

# Metadata
- Mangled symbol: `solve_linear_system_lu`
- Type safety:  From compilation
"""

function solve_linear_system_lu(lu::Ptr{LUDecomposition}, b::Ptr{Cdouble}, x::Ptr{Cdouble}, n::Csize_t)::Status
    return ccall((:solve_linear_system_lu, LIBRARY_PATH), Status, (Ptr{LUDecomposition}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), lu, b, x, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_linear_system_lu(lu::LUDecomposition, b::Ptr{Cdouble}, x::Vector{Float64}, n::Csize_t)::Any
    return GC.@preserve x begin
        ccall((:solve_linear_system_lu, LIBRARY_PATH), Any, (Ptr{LUDecomposition}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), Ref(lu), b, pointer(x), n)
    end
end

"""
    solve_linear_system_qr(qr::Ptr{QRDecomposition}, b::Ptr{Cdouble}, x::Ptr{Cdouble}) -> Any

Wrapper for C++ function: `solve_linear_system_qr`

# Arguments
- `qr::Ptr{QRDecomposition}`
- `b::Ptr{Cdouble}`
- `x::Ptr{Cdouble}`

# Returns
- `Any`

# Metadata
- Mangled symbol: `solve_linear_system_qr`
- Type safety:  From compilation
"""

function solve_linear_system_qr(qr::Ptr{QRDecomposition}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Status
    return ccall((:solve_linear_system_qr, LIBRARY_PATH), Status, (Ptr{QRDecomposition}, Ptr{Cdouble}, Ptr{Cdouble},), qr, b, x)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_linear_system_qr(qr::QRDecomposition, b::Ptr{Cdouble}, x::Vector{Float64})::Any
    return GC.@preserve x begin
        ccall((:solve_linear_system_qr, LIBRARY_PATH), Any, (Ptr{QRDecomposition}, Ptr{Cdouble}, Ptr{Cdouble},), Ref(qr), b, pointer(x))
    end
end

"""
    solve_ode_adaptive(ode_func::Ptr{Cvoid}, t0::Cdouble, t_final::Cdouble, y0::Ptr{Cdouble}, n::Csize_t, tolerance::Cdouble, event_func::Ptr{Cvoid}, user_data::Ptr{Cvoid}) -> ODEResult

Wrapper for C++ function: `solve_ode_adaptive`

# Arguments
- `ode_func::Ptr{Cvoid}` - Callback function
- `t0::Cdouble`
- `t_final::Cdouble`
- `y0::Ptr{Cdouble}`
- `n::Csize_t`
- `tolerance::Cdouble`
- `event_func::Ptr{Cvoid}` - Callback function
- `user_data::Ptr{Cvoid}`

# Returns
- `ODEResult`

            # Callback Signatures
**Callback `ode_func`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

**Callback `event_func`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `solve_ode_adaptive`
- Type safety:  From compilation
"""

function solve_ode_adaptive(ode_func::Ptr{Cvoid}, t0::Cdouble, t_final::Cdouble, y0::Ptr{Cdouble}, n::Csize_t, tolerance::Cdouble, event_func::Ptr{Cvoid}, user_data::Ptr{Cvoid})::ODEResult
    ccall((:solve_ode_adaptive, LIBRARY_PATH), ODEResult, (Ptr{Cvoid}, Cdouble, Cdouble, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid}, Ptr{Cvoid},), ode_func, t0, t_final, y0, n, tolerance, event_func, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_ode_adaptive(ode_func::Ptr{Cvoid}, t0::Cdouble, t_final::Cdouble, y0::Vector{Float64}, n::Csize_t, tolerance::Cdouble, event_func::Ptr{Cvoid}, user_data::Ptr{Cvoid})::ODEResult
    return GC.@preserve y0 begin
        ccall((:solve_ode_adaptive, LIBRARY_PATH), ODEResult, (Ptr{Cvoid}, Cdouble, Cdouble, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid}, Ptr{Cvoid},), ode_func, t0, t_final, pointer(y0), n, tolerance, event_func, user_data)
    end
end

"""
    solve_ode_rk4(ode_func::Ptr{Cvoid}, t0::Cdouble, t_final::Cdouble, y0::Ptr{Cdouble}, n::Csize_t, dt::Cdouble, user_data::Ptr{Cvoid}) -> ODEResult

Wrapper for C++ function: `solve_ode_rk4`

# Arguments
- `ode_func::Ptr{Cvoid}` - Callback function
- `t0::Cdouble`
- `t_final::Cdouble`
- `y0::Ptr{Cdouble}`
- `n::Csize_t`
- `dt::Cdouble`
- `user_data::Ptr{Cvoid}`

# Returns
- `ODEResult`

            # Callback Signatures
**Callback `ode_func`**: Create using `@cfunction`
```julia
callback = @cfunction(my_callback, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cvoid},)) Ptr{Cvoid}
```

# Metadata
- Mangled symbol: `solve_ode_rk4`
- Type safety:  From compilation
"""

function solve_ode_rk4(ode_func::Ptr{Cvoid}, t0::Cdouble, t_final::Cdouble, y0::Ptr{Cdouble}, n::Csize_t, dt::Cdouble, user_data::Ptr{Cvoid})::ODEResult
    ccall((:solve_ode_rk4, LIBRARY_PATH), ODEResult, (Ptr{Cvoid}, Cdouble, Cdouble, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid},), ode_func, t0, t_final, y0, n, dt, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_ode_rk4(ode_func::Ptr{Cvoid}, t0::Cdouble, t_final::Cdouble, y0::Vector{Float64}, n::Csize_t, dt::Cdouble, user_data::Ptr{Cvoid})::ODEResult
    return GC.@preserve y0 begin
        ccall((:solve_ode_rk4, LIBRARY_PATH), ODEResult, (Ptr{Cvoid}, Cdouble, Cdouble, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid},), ode_func, t0, t_final, pointer(y0), n, dt, user_data)
    end
end

"""
    sparse_matrix_create(rows::Csize_t, cols::Csize_t, nnz::Csize_t) -> SparseMatrix

Wrapper for C++ function: `sparse_matrix_create`

# Arguments
- `rows::Csize_t`
- `cols::Csize_t`
- `nnz::Csize_t`

# Returns
- `SparseMatrix`

# Metadata
- Mangled symbol: `sparse_matrix_create`
- Type safety:  From compilation
"""

function sparse_matrix_create(rows::Csize_t, cols::Csize_t, nnz::Csize_t)::SparseMatrix
    ccall((:sparse_matrix_create, LIBRARY_PATH), SparseMatrix, (Csize_t, Csize_t, Csize_t,), rows, cols, nnz)
end

"""
    sparse_matrix_destroy(mat::Ptr{SparseMatrix}) -> Cvoid

Wrapper for C++ function: `sparse_matrix_destroy`

# Arguments
- `mat::Ptr{SparseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `sparse_matrix_destroy`
- Type safety:  From compilation
"""

function sparse_matrix_destroy(mat::Ptr{SparseMatrix})::Cvoid
    ccall((:sparse_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{SparseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function sparse_matrix_destroy(mat::SparseMatrix)::Cvoid
    return ccall((:sparse_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{SparseMatrix},), Ref(mat))
end

"""
    spline_destroy(spline::Ptr{SplineInterpolation}) -> Cvoid

Wrapper for C++ function: `spline_destroy`

# Arguments
- `spline::Ptr{SplineInterpolation}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `spline_destroy`
- Type safety:  From compilation
"""

function spline_destroy(spline::Ptr{SplineInterpolation})::Cvoid
    ccall((:spline_destroy, LIBRARY_PATH), Cvoid, (Ptr{SplineInterpolation},), spline)
end

"""
    spline_eval(spline::Ptr{SplineInterpolation}, x::Cdouble) -> Cdouble

Wrapper for C++ function: `spline_eval`

# Arguments
- `spline::Ptr{SplineInterpolation}`
- `x::Cdouble`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `spline_eval`
- Type safety:  From compilation
"""

function spline_eval(spline::Ptr{SplineInterpolation}, x::Cdouble)::Cdouble
    ccall((:spline_eval, LIBRARY_PATH), Cdouble, (Ptr{SplineInterpolation}, Cdouble,), spline, x)
end

"""
    status_to_string(status::Status) -> Cstring

Wrapper for C++ function: `status_to_string`

# Arguments
- `status::Status`

# Returns
- `Cstring`

# Metadata
- Mangled symbol: `status_to_string`
- Type safety:  From compilation
"""

function status_to_string(status::Status)::String
    ptr = ccall((:status_to_string, LIBRARY_PATH), Cstring, (Status,), status)
    if ptr == C_NULL
        error("status_to_string returned NULL pointer")
    end
    return unsafe_string(ptr)
end

"""
    vector_axpy(y::Ptr{Cdouble}, alpha::Cdouble, x::Ptr{Cdouble}, n::Csize_t) -> Cvoid

Wrapper for C++ function: `vector_axpy`

# Arguments
- `y::Ptr{Cdouble}`
- `alpha::Cdouble`
- `x::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `vector_axpy`
- Type safety:  From compilation
"""

function vector_axpy(y::Ptr{Cdouble}, alpha::Cdouble, x::Ptr{Cdouble}, n::Csize_t)::Cvoid
    ccall((:vector_axpy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Ptr{Cdouble}, Csize_t,), y, alpha, x, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_axpy(y::Vector{Float64}, alpha::Cdouble, x::Vector{Float64}, n::Csize_t)::Cvoid
    return GC.@preserve y x begin
        ccall((:vector_axpy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Ptr{Cdouble}, Csize_t,), pointer(y), alpha, pointer(x), n)
    end
end

"""
    vector_copy(dest::Ptr{Cdouble}, src::Ptr{Cdouble}, n::Csize_t) -> Cvoid

Wrapper for C++ function: `vector_copy`

# Arguments
- `dest::Ptr{Cdouble}`
- `src::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `vector_copy`
- Type safety:  From compilation
"""

function vector_copy(dest::Ptr{Cdouble}, src::Ptr{Cdouble}, n::Csize_t)::Cvoid
    ccall((:vector_copy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), dest, src, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_copy(dest::Ptr{Cdouble}, src::Vector{Float64}, n::Csize_t)::Cvoid
    return GC.@preserve src begin
        ccall((:vector_copy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), dest, pointer(src), n)
    end
end

"""
    vector_dot(x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t) -> Cdouble

Wrapper for C++ function: `vector_dot`

# Arguments
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `vector_dot`
- Type safety:  From compilation
"""

function vector_dot(x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t)::Cdouble
    ccall((:vector_dot, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), x, y, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_dot(x::Vector{Float64}, y::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve x y begin
        ccall((:vector_dot, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), pointer(x), pointer(y), n)
    end
end

"""
    vector_norm(x::Ptr{Cdouble}, n::Csize_t) -> Cdouble

Wrapper for C++ function: `vector_norm`

# Arguments
- `x::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `vector_norm`
- Type safety:  From compilation
"""

function vector_norm(x::Ptr{Cdouble}, n::Csize_t)::Cdouble
    ccall((:vector_norm, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), x, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_norm(x::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve x begin
        ccall((:vector_norm, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(x), n)
    end
end

"""
    vector_scale(x::Ptr{Cdouble}, alpha::Cdouble, n::Csize_t) -> Cvoid

Wrapper for C++ function: `vector_scale`

# Arguments
- `x::Ptr{Cdouble}`
- `alpha::Cdouble`
- `n::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `vector_scale`
- Type safety:  From compilation
"""

function vector_scale(x::Ptr{Cdouble}, alpha::Cdouble, n::Csize_t)::Cvoid
    ccall((:vector_scale, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Csize_t,), x, alpha, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_scale(x::Vector{Float64}, alpha::Cdouble, n::Csize_t)::Cvoid
    return GC.@preserve x begin
        ccall((:vector_scale, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Csize_t,), pointer(x), alpha, n)
    end
end


end # module StressTest
