# Auto-generated Julia wrapper for stress_test
# Generated: 2026-02-04 03:03:43
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libstress_test.so
# Metadata: compilation_metadata.json

module StressTest

using Libdl
import RepliBuild

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/stress_test/julia/libstress_test.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

function __init__()
    # Initialize the global JIT context with this library's vtables
    RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)
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
    "generated_at" => "2026-02-04T03:03:43.762"
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
struct DenseMatrix
    data::Ptr{Cdouble}
    rows::Csize_t
    cols::Csize_t
    owns_data::Bool
end

# C++ struct: FFTResult (3 members)
struct FFTResult
    real::Ptr{Cdouble}
    imag::Ptr{Cdouble}
    n::Csize_t
end

# C++ struct: Histogram (3 members)
struct Histogram
    bin_edges::Ptr{Cdouble}
    counts::Ptr{Int32}
    n_bins::Csize_t
end

# C++ struct: ODEResult (6 members)
struct ODEResult
    y::Ptr{Cdouble}
    t_values::Ptr{Cdouble}
    y_values::Ptr{Ptr{Cdouble}}
    n_steps::Csize_t
    dimension::Csize_t
    status::Status
end

# C++ struct: OptimizationOptions (6 members)
struct OptimizationOptions
    tolerance::Cdouble
    step_size::Cdouble
    max_iterations::Int32
    max_function_evals::Int32
    algorithm::OptimizationAlgorithm
    verbose::Bool
end

# C++ struct: OptimizationState (8 members)
struct OptimizationState
    x::Ptr{Cdouble}
    gradient::Ptr{Cdouble}
    f_value::Cdouble
    gradient_norm::Cdouble
    iteration::Int32
    n_evals::Int32
    status::Status
    dimension::Csize_t
end

# C++ struct: Polynomial (2 members)
struct Polynomial
    coefficients::Ptr{Cdouble}
    degree::Csize_t
end

# C++ struct: SparseMatrix (6 members)
struct SparseMatrix
    values::Ptr{Cdouble}
    row_indices::Ptr{Int32}
    col_pointers::Ptr{Int32}
    nnz::Csize_t
    rows::Csize_t
    cols::Csize_t
end

# C++ struct: SplineInterpolation (5 members)
struct SplineInterpolation
    x_points::Ptr{Cdouble}
    y_points::Ptr{Cdouble}
    coefficients::Ptr{Cdouble}
    n_points::Csize_t
    n_coeffs::Csize_t
end

# C++ struct: __va_list_tag (4 members)
struct __va_list_tag
    gp_offset::Cuint
    fp_offset::Cuint
    overflow_arg_area::Ptr{Cvoid}
    reg_save_area::Ptr{Cvoid}
end

# C++ struct: mersenne_twister_engine<unsigned long, 64UL, 312UL, 156UL, 31UL, 13043109905998158313UL, 29UL, 6148914691236517205UL, 17UL, 8202884508482404352UL, 37UL, 18444473444759240704UL, 43UL, 6364136223846793005UL> (2 members)
struct mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL
    _M_x::NTuple{312, Culong}
    _M_p::Csize_t
end

# C++ struct: mersenne_twister_engine<unsigned long, 64UL, 312UL, 156UL, 31UL, 13043109905998158313UL, 29UL, 6148914691236517205UL, 17UL, 8202884508482404352UL, 37UL, 18444473444759240704UL, 43UL, 6364136223846793005UL>, double> (1 members)
struct mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL_double
    _M_g::Ref{mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL}
end

# C++ struct: param_type (4 members)
struct param_type
    _M_mean::Cdouble
    _M_stddev::Cdouble
    _M_saved::Cdouble
    _M_saved_available::Bool
end

# C++ struct: EigenDecomposition (5 members)
struct EigenDecomposition
    eigenvalues::Ptr{Cdouble}
    eigenvalues_imag::Ptr{Cdouble}
    eigenvectors::DenseMatrix
    n::Csize_t
    status::Status
end

# C++ struct: LUDecomposition (5 members)
struct LUDecomposition
    L::DenseMatrix
    U::DenseMatrix
    permutation::Ptr{Int32}
    size::Csize_t
    status::Status
end

# C++ struct: QRDecomposition (5 members)
struct QRDecomposition
    Q::DenseMatrix
    R::DenseMatrix
    m::Csize_t
    n::Csize_t
    status::Status
end

# C++ struct: normal_distribution<double> (1 members)
struct normal_distribution_double
    _M_param::param_type
end

# C++ struct: uniform_real_distribution<double> (1 members)
struct uniform_real_distribution_double
    _M_param::param_type
end


# =============================================================================
# Managed Types (Auto-Finalizers)
# =============================================================================

mutable struct ManagedFFTResult
    handle::Ptr{FFTResult}
    
    function ManagedFFTResult(ptr::Ptr{FFTResult})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedFFTResult")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: fft_result_destroy(x.handle)
            ccall((:fft_result_destroy, LIBRARY_PATH), Cvoid, (Ptr{FFTResult},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{FFTResult}}, obj::ManagedFFTResult) = obj.handle

export ManagedFFTResult

mutable struct ManagedODEResult
    handle::Ptr{ODEResult}
    
    function ManagedODEResult(ptr::Ptr{ODEResult})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedODEResult")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: ode_result_destroy(x.handle)
            ccall((:ode_result_destroy, LIBRARY_PATH), Cvoid, (Ptr{ODEResult},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{ODEResult}}, obj::ManagedODEResult) = obj.handle

export ManagedODEResult

mutable struct ManagedSplineInterpolation
    handle::Ptr{SplineInterpolation}
    
    function ManagedSplineInterpolation(ptr::Ptr{SplineInterpolation})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedSplineInterpolation")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: spline_destroy(x.handle)
            ccall((:spline_destroy, LIBRARY_PATH), Cvoid, (Ptr{SplineInterpolation},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{SplineInterpolation}}, obj::ManagedSplineInterpolation) = obj.handle

export ManagedSplineInterpolation

mutable struct ManagedDenseMatrix
    handle::Ptr{DenseMatrix}
    
    function ManagedDenseMatrix(ptr::Ptr{DenseMatrix})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedDenseMatrix")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: dense_matrix_destroy(x.handle)
            ccall((:dense_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{DenseMatrix}}, obj::ManagedDenseMatrix) = obj.handle

export ManagedDenseMatrix

mutable struct ManagedHistogram
    handle::Ptr{Histogram}
    
    function ManagedHistogram(ptr::Ptr{Histogram})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedHistogram")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: histogram_destroy(x.handle)
            ccall((:histogram_destroy, LIBRARY_PATH), Cvoid, (Ptr{Histogram},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{Histogram}}, obj::ManagedHistogram) = obj.handle

export ManagedHistogram

mutable struct ManagedPolynomial
    handle::Ptr{Polynomial}
    
    function ManagedPolynomial(ptr::Ptr{Polynomial})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedPolynomial")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: polynomial_destroy(x.handle)
            ccall((:polynomial_destroy, LIBRARY_PATH), Cvoid, (Ptr{Polynomial},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{Polynomial}}, obj::ManagedPolynomial) = obj.handle

export ManagedPolynomial

mutable struct ManagedSparseMatrix
    handle::Ptr{SparseMatrix}
    
    function ManagedSparseMatrix(ptr::Ptr{SparseMatrix})
        if ptr == C_NULL
            error("Cannot wrap NULL pointer in ManagedSparseMatrix")
        end
        obj = new(ptr)
        finalizer(obj) do x
            # Call deleter: sparse_matrix_destroy(x.handle)
            ccall((:sparse_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{SparseMatrix},), x.handle)
        end
        return obj
    end
end

# Allow passing Managed object to ccall expecting Ptr
Base.unsafe_convert(::Type{Ptr{SparseMatrix}}, obj::ManagedSparseMatrix) = obj.handle

export ManagedSparseMatrix

export compute_eigen, compute_fft, compute_histogram, compute_ifft, compute_lu, compute_mean, compute_median, compute_qr, compute_quantiles, compute_stddev, compute_variance, convolve, correlate, create_cubic_spline, dense_matrix_copy, dense_matrix_create, dense_matrix_destroy, dense_matrix_resize, dense_matrix_set_identity, dense_matrix_set_zero, fft_result_destroy, fill_random_normal, fill_random_uniform, histogram_destroy, line_search_backtracking, matrix_add, matrix_determinant, matrix_multiply, matrix_trace, matrix_transpose, matrix_vector_mult, matrix_vector_mult_add, ode_result_destroy, optimize_minimize, optimize_minimize_numerical_gradient, polynomial_destroy, polynomial_eval, polynomial_fit, print_matrix, print_vector, set_random_seed, solve_conjugate_gradient, solve_least_squares, solve_linear_system_lu, solve_linear_system_qr, solve_ode_adaptive, solve_ode_rk4, sparse_matrix_create, sparse_matrix_destroy, spline_destroy, spline_eval, status_to_string, vector_axpy, vector_copy, vector_dot, vector_norm, vector_scale, OptimizationAlgorithm, GRADIENT_DESCENT, CONJUGATE_GRADIENT, LBFGS, NEWTON, Status, SUCCESS, ERROR_INVALID_INPUT, ERROR_SINGULAR_MATRIX, ERROR_NOT_CONVERGED, ERROR_OUT_OF_MEMORY, ERROR_DIMENSION_MISMATCH, LUDecomposition, ODEResult, EigenDecomposition, DenseMatrix, SparseMatrix, mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL, QRDecomposition, uniform_real_distribution_double, FFTResult, mersenne_twister_engine_unsignedlong_64UL_312UL_156UL_31UL_13043109905998158313UL_29UL_6148914691236517205UL_17UL_8202884508482404352UL_37UL_18444473444759240704UL_43UL_6364136223846793005UL_double, __va_list_tag, OptimizationOptions, Histogram, normal_distribution_double, Polynomial, OptimizationState, param_type, SplineInterpolation

"""
    compute_eigen(A::Any) -> EigenDecomposition

Wrapper for C++ function: `compute_eigen`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `EigenDecomposition`

# Metadata
- Mangled symbol: `compute_eigen`
"""

function compute_eigen(A::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_compute_eigen_thunk", A)
end
"""
    compute_fft(signal::Any, n::Csize_t) -> FFTResult

Wrapper for C++ function: `compute_fft`

# Arguments
- `signal::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `FFTResult`

# Metadata
- Mangled symbol: `compute_fft`
"""

function compute_fft(signal::Any, n::Csize_t)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_compute_fft_thunk", signal, n)
end
"""
    compute_histogram(data::Any, n::Csize_t, n_bins::Csize_t, min_val::Cdouble, max_val::Cdouble) -> Histogram

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
"""

function compute_histogram(data::Any, n::Csize_t, n_bins::Csize_t, min_val::Cdouble, max_val::Cdouble)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_compute_histogram_thunk", data, n, n_bins, min_val, max_val)
end
"""
    compute_ifft(fft_data::Any, signal_out::Any) -> Cvoid

Wrapper for C++ function: `compute_ifft`

# Arguments
- `fft_data::Ptr{FFTResult}`
- `signal_out::Ptr{Cdouble}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `compute_ifft`
"""

function compute_ifft(fft_data::Any, signal_out::Any)::Cvoid
    ccall((:compute_ifft, LIBRARY_PATH), Cvoid, (Ptr{FFTResult}, Ptr{Cdouble},), fft_data, signal_out)
end

# Convenience wrapper - accepts structs directly instead of pointers
function compute_ifft(fft_data::FFTResult, signal_out::Ptr{Cdouble})::Cvoid
    return ccall((:compute_ifft, LIBRARY_PATH), Cvoid, (Ptr{FFTResult}, Ptr{Cdouble},), Ref(fft_data), signal_out)
end

"""
    compute_lu(A::Any) -> LUDecomposition

Wrapper for C++ function: `compute_lu`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `LUDecomposition`

# Metadata
- Mangled symbol: `compute_lu`
"""

function compute_lu(A::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_compute_lu_thunk", A)
end
"""
    compute_mean(data::Any, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_mean`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_mean`
"""

function compute_mean(data::Any, n::Csize_t)::Cdouble
    ccall((:compute_mean, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_mean(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_mean, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    compute_median(data::Any, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_median`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_median`
"""

function compute_median(data::Any, n::Csize_t)::Cdouble
    ccall((:compute_median, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_median(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_median, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    compute_qr(A::Any) -> QRDecomposition

Wrapper for C++ function: `compute_qr`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `QRDecomposition`

# Metadata
- Mangled symbol: `compute_qr`
"""

function compute_qr(A::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_compute_qr_thunk", A)
end
"""
    compute_quantiles(data::Any, n::Csize_t, probabilities::Any, quantiles::Any, n_quantiles::Csize_t) -> Cvoid

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
"""

function compute_quantiles(data::Any, n::Csize_t, probabilities::Any, quantiles::Any, n_quantiles::Csize_t)::Cvoid
    ccall((:compute_quantiles, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), data, n, probabilities, quantiles, n_quantiles)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_quantiles(data::Vector{Float64}, n::Csize_t, probabilities::Ptr{Cdouble}, quantiles::Ptr{Cdouble}, n_quantiles::Csize_t)::Cvoid
    return GC.@preserve data begin
        ccall((:compute_quantiles, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), pointer(data), n, probabilities, quantiles, n_quantiles)
    end
end

"""
    compute_stddev(data::Any, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_stddev`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_stddev`
"""

function compute_stddev(data::Any, n::Csize_t)::Cdouble
    ccall((:compute_stddev, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_stddev(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_stddev, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    compute_variance(data::Any, n::Csize_t) -> Cdouble

Wrapper for C++ function: `compute_variance`

# Arguments
- `data::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `compute_variance`
"""

function compute_variance(data::Any, n::Csize_t)::Cdouble
    ccall((:compute_variance, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), data, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function compute_variance(data::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve data begin
        ccall((:compute_variance, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(data), n)
    end
end

"""
    convolve(signal1::Any, n1::Csize_t, signal2::Any, n2::Csize_t, result::Any) -> Cvoid

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
"""

function convolve(signal1::Any, n1::Csize_t, signal2::Any, n2::Csize_t, result::Any)::Cvoid
    ccall((:convolve, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), signal1, n1, signal2, n2, result)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function convolve(signal1::Vector{Float64}, n1::Csize_t, signal2::Vector{Float64}, n2::Csize_t, result::Ptr{Cdouble})::Cvoid
    return GC.@preserve signal1 signal2 begin
        ccall((:convolve, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), pointer(signal1), n1, pointer(signal2), n2, result)
    end
end

"""
    correlate(signal1::Any, signal2::Any, n::Csize_t, result::Any) -> Cvoid

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
"""

function correlate(signal1::Any, signal2::Any, n::Csize_t, result::Any)::Cvoid
    ccall((:correlate, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), signal1, signal2, n, result)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function correlate(signal1::Vector{Float64}, signal2::Vector{Float64}, n::Csize_t, result::Ptr{Cdouble})::Cvoid
    return GC.@preserve signal1 signal2 begin
        ccall((:correlate, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble},), pointer(signal1), pointer(signal2), n, result)
    end
end

"""
    create_cubic_spline(x::Any, y::Any, n::Csize_t) -> SplineInterpolation

Wrapper for C++ function: `create_cubic_spline`

# Arguments
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `SplineInterpolation`

# Metadata
- Mangled symbol: `create_cubic_spline`
"""

function create_cubic_spline(x::Any, y::Any, n::Csize_t)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_create_cubic_spline_thunk", x, y, n)
end
"""
    dense_matrix_copy(src::Any) -> DenseMatrix

Wrapper for C++ function: `dense_matrix_copy`

# Arguments
- `src::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `dense_matrix_copy`
"""

function dense_matrix_copy(src::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_dense_matrix_copy_thunk", src)
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
"""

function dense_matrix_create(rows::Csize_t, cols::Csize_t)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_dense_matrix_create_thunk", rows, cols)
end
"""
    dense_matrix_destroy(mat::Any) -> Cvoid

Wrapper for C++ function: `dense_matrix_destroy`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `dense_matrix_destroy`
"""

function dense_matrix_destroy(mat::Any)::Cvoid
    ccall((:dense_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_destroy(mat::DenseMatrix)::Cvoid
    return ccall((:dense_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    dense_matrix_resize(mat::Any, new_rows::Csize_t, new_cols::Csize_t) -> Any

Wrapper for C++ function: `dense_matrix_resize`

# Arguments
- `mat::Ptr{DenseMatrix}`
- `new_rows::Csize_t`
- `new_cols::Csize_t`

# Returns
- `Any`

# Metadata
- Mangled symbol: `dense_matrix_resize`
"""

function dense_matrix_resize(mat::Any, new_rows::Csize_t, new_cols::Csize_t)::Status
    return ccall((:dense_matrix_resize, LIBRARY_PATH), Status, (Ptr{DenseMatrix}, Csize_t, Csize_t,), mat, new_rows, new_cols)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_resize(mat::DenseMatrix, new_rows::Csize_t, new_cols::Csize_t)::Any
    return ccall((:dense_matrix_resize, LIBRARY_PATH), Any, (Ptr{DenseMatrix}, Csize_t, Csize_t,), Ref(mat), new_rows, new_cols)
end

"""
    dense_matrix_set_identity(mat::Any) -> Cvoid

Wrapper for C++ function: `dense_matrix_set_identity`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `dense_matrix_set_identity`
"""

function dense_matrix_set_identity(mat::Any)::Cvoid
    ccall((:dense_matrix_set_identity, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_set_identity(mat::DenseMatrix)::Cvoid
    return ccall((:dense_matrix_set_identity, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    dense_matrix_set_zero(mat::Any) -> Cvoid

Wrapper for C++ function: `dense_matrix_set_zero`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `dense_matrix_set_zero`
"""

function dense_matrix_set_zero(mat::Any)::Cvoid
    ccall((:dense_matrix_set_zero, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function dense_matrix_set_zero(mat::DenseMatrix)::Cvoid
    return ccall((:dense_matrix_set_zero, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    fft_result_destroy(result::Any) -> Cvoid

Wrapper for C++ function: `fft_result_destroy`

# Arguments
- `result::Ptr{FFTResult}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `fft_result_destroy`
"""

function fft_result_destroy(result::Any)::Cvoid
    ccall((:fft_result_destroy, LIBRARY_PATH), Cvoid, (Ptr{FFTResult},), result)
end

# Convenience wrapper - accepts structs directly instead of pointers
function fft_result_destroy(result::FFTResult)::Cvoid
    return ccall((:fft_result_destroy, LIBRARY_PATH), Cvoid, (Ptr{FFTResult},), Ref(result))
end

"""
    fill_random_normal(data::Any, n::Csize_t, mean::Cdouble, stddev::Cdouble) -> Cvoid

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
"""

function fill_random_normal(data::Any, n::Csize_t, mean::Cdouble, stddev::Cdouble)::Cvoid
    ccall((:fill_random_normal, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), data, n, mean, stddev)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function fill_random_normal(data::Vector{Float64}, n::Csize_t, mean::Cdouble, stddev::Cdouble)::Cvoid
    return GC.@preserve data begin
        ccall((:fill_random_normal, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), pointer(data), n, mean, stddev)
    end
end

"""
    fill_random_uniform(data::Any, n::Csize_t, min_val::Cdouble, max_val::Cdouble) -> Cvoid

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
"""

function fill_random_uniform(data::Any, n::Csize_t, min_val::Cdouble, max_val::Cdouble)::Cvoid
    ccall((:fill_random_uniform, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), data, n, min_val, max_val)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function fill_random_uniform(data::Vector{Float64}, n::Csize_t, min_val::Cdouble, max_val::Cdouble)::Cvoid
    return GC.@preserve data begin
        ccall((:fill_random_uniform, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t, Cdouble, Cdouble,), pointer(data), n, min_val, max_val)
    end
end

"""
    histogram_destroy(hist::Any) -> Cvoid

Wrapper for C++ function: `histogram_destroy`

# Arguments
- `hist::Ptr{Histogram}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `histogram_destroy`
"""

function histogram_destroy(hist::Any)::Cvoid
    ccall((:histogram_destroy, LIBRARY_PATH), Cvoid, (Ptr{Histogram},), hist)
end

# Convenience wrapper - accepts structs directly instead of pointers
function histogram_destroy(hist::Histogram)::Cvoid
    return ccall((:histogram_destroy, LIBRARY_PATH), Cvoid, (Ptr{Histogram},), Ref(hist))
end

"""
    line_search_backtracking(objective::Any, x::Any, direction::Any, x_new::Any, n::Csize_t, initial_step::Cdouble, user_data::Any) -> Cdouble

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
"""

function line_search_backtracking(objective::Any, x::Any, direction::Any, x_new::Any, n::Csize_t, initial_step::Cdouble, user_data::Any)::Cdouble
    ccall((:line_search_backtracking, LIBRARY_PATH), Cdouble, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid},), objective, x, direction, x_new, n, initial_step, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function line_search_backtracking(objective::Ptr{Cvoid}, x::Vector{Float64}, direction::Ptr{Cdouble}, x_new::Vector{Float64}, n::Csize_t, initial_step::Cdouble, user_data::Ptr{Cvoid})::Cdouble
    return GC.@preserve x x_new begin
        ccall((:line_search_backtracking, LIBRARY_PATH), Cdouble, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid},), objective, pointer(x), direction, pointer(x_new), n, initial_step, user_data)
    end
end

"""
    matrix_add(A::Any, B::Any) -> DenseMatrix

Wrapper for C++ function: `matrix_add`

# Arguments
- `A::Ptr{DenseMatrix}`
- `B::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `matrix_add`
"""

function matrix_add(A::Any, B::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_matrix_add_thunk", A, B)
end
"""
    matrix_determinant(A::Any) -> Cdouble

Wrapper for C++ function: `matrix_determinant`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `matrix_determinant`
"""

function matrix_determinant(A::Any)::Cdouble
    ccall((:matrix_determinant, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function matrix_determinant(A::DenseMatrix)::Cdouble
    return ccall((:matrix_determinant, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), Ref(A))
end

"""
    matrix_multiply(A::Any, B::Any) -> DenseMatrix

Wrapper for C++ function: `matrix_multiply`

# Arguments
- `A::Ptr{DenseMatrix}`
- `B::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `matrix_multiply`
"""

function matrix_multiply(A::Any, B::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_matrix_multiply_thunk", A, B)
end
"""
    matrix_trace(A::Any) -> Cdouble

Wrapper for C++ function: `matrix_trace`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `matrix_trace`
"""

function matrix_trace(A::Any)::Cdouble
    ccall((:matrix_trace, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), A)
end

# Convenience wrapper - accepts structs directly instead of pointers
function matrix_trace(A::DenseMatrix)::Cdouble
    return ccall((:matrix_trace, LIBRARY_PATH), Cdouble, (Ptr{DenseMatrix},), Ref(A))
end

"""
    matrix_transpose(A::Any) -> DenseMatrix

Wrapper for C++ function: `matrix_transpose`

# Arguments
- `A::Ptr{DenseMatrix}`

# Returns
- `DenseMatrix`

# Metadata
- Mangled symbol: `matrix_transpose`
"""

function matrix_transpose(A::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_matrix_transpose_thunk", A)
end
"""
    matrix_vector_mult(A::Any, x::Any, y::Any) -> Cvoid

Wrapper for C++ function: `matrix_vector_mult`

# Arguments
- `A::Ptr{DenseMatrix}`
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `matrix_vector_mult`
"""

function matrix_vector_mult(A::Any, x::Any, y::Any)::Cvoid
    ccall((:matrix_vector_mult, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), A, x, y)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function matrix_vector_mult(A::DenseMatrix, x::Vector{Float64}, y::Vector{Float64})::Cvoid
    return GC.@preserve x y begin
        ccall((:matrix_vector_mult, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), Ref(A), pointer(x), pointer(y))
    end
end

"""
    matrix_vector_mult_add(A::Any, x::Any, y::Any, alpha::Cdouble, beta::Cdouble) -> Cvoid

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
"""

function matrix_vector_mult_add(A::Any, x::Any, y::Any, alpha::Cdouble, beta::Cdouble)::Cvoid
    ccall((:matrix_vector_mult_add, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,), A, x, y, alpha, beta)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function matrix_vector_mult_add(A::DenseMatrix, x::Vector{Float64}, y::Vector{Float64}, alpha::Cdouble, beta::Cdouble)::Cvoid
    return GC.@preserve x y begin
        ccall((:matrix_vector_mult_add, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,), Ref(A), pointer(x), pointer(y), alpha, beta)
    end
end

"""
    ode_result_destroy(result::Any) -> Cvoid

Wrapper for C++ function: `ode_result_destroy`

# Arguments
- `result::Ptr{ODEResult}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `ode_result_destroy`
"""

function ode_result_destroy(result::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_ode_result_destroy_thunk", result)
end
"""
    optimize_minimize(objective::Any, gradient::Any, x::Any, n::Csize_t, options::Any, final_state::Any, callback::Any, user_data::Any) -> Any

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
"""

function optimize_minimize(objective::Any, gradient::Any, x::Any, n::Csize_t, options::Any, final_state::Any, callback::Any, user_data::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_optimize_minimize_thunk", objective, gradient, x, n, options, final_state, callback, user_data)
end
"""
    optimize_minimize_numerical_gradient(objective::Any, x::Any, n::Csize_t, options::Any, final_state::Any, callback::Any, user_data::Any) -> Any

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
"""

function optimize_minimize_numerical_gradient(objective::Any, x::Any, n::Csize_t, options::Any, final_state::Any, callback::Any, user_data::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_optimize_minimize_numerical_gradient_thunk", objective, x, n, options, final_state, callback, user_data)
end
"""
    polynomial_destroy(poly::Any) -> Cvoid

Wrapper for C++ function: `polynomial_destroy`

# Arguments
- `poly::Ptr{Polynomial}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `polynomial_destroy`
"""

function polynomial_destroy(poly::Any)::Cvoid
    ccall((:polynomial_destroy, LIBRARY_PATH), Cvoid, (Ptr{Polynomial},), poly)
end

"""
    polynomial_eval(poly::Any, x::Cdouble) -> Cdouble

Wrapper for C++ function: `polynomial_eval`

# Arguments
- `poly::Ptr{Polynomial}`
- `x::Cdouble`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `polynomial_eval`
"""

function polynomial_eval(poly::Any, x::Cdouble)::Cdouble
    ccall((:polynomial_eval, LIBRARY_PATH), Cdouble, (Ptr{Polynomial}, Cdouble,), poly, x)
end

"""
    polynomial_fit(x::Any, y::Any, n::Csize_t, degree::Csize_t) -> Polynomial

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
"""

function polynomial_fit(x::Any, y::Any, n::Csize_t, degree::Csize_t)::Polynomial
    ccall((:polynomial_fit, LIBRARY_PATH), Polynomial, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Csize_t,), x, y, n, degree)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function polynomial_fit(x::Vector{Float64}, y::Vector{Float64}, n::Csize_t, degree::Csize_t)::Polynomial
    return GC.@preserve x y begin
        ccall((:polynomial_fit, LIBRARY_PATH), Polynomial, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Csize_t,), pointer(x), pointer(y), n, degree)
    end
end

"""
    print_matrix(mat::Any) -> Cvoid

Wrapper for C++ function: `print_matrix`

# Arguments
- `mat::Ptr{DenseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `print_matrix`
"""

function print_matrix(mat::Any)::Cvoid
    ccall((:print_matrix, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function print_matrix(mat::DenseMatrix)::Cvoid
    return ccall((:print_matrix, LIBRARY_PATH), Cvoid, (Ptr{DenseMatrix},), Ref(mat))
end

"""
    print_vector(vec::Any, n::Csize_t) -> Cvoid

Wrapper for C++ function: `print_vector`

# Arguments
- `vec::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `print_vector`
"""

function print_vector(vec::Any, n::Csize_t)::Cvoid
    ccall((:print_vector, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t,), vec, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function print_vector(vec::Vector{Float64}, n::Csize_t)::Cvoid
    return GC.@preserve vec begin
        ccall((:print_vector, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Csize_t,), pointer(vec), n)
    end
end

"""
    set_random_seed(seed::UInt64) -> Cvoid

Wrapper for C++ function: `set_random_seed`

# Arguments
- `seed::UInt64`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `set_random_seed`
"""

function set_random_seed(seed::UInt64)::Cvoid
    ccall((:set_random_seed, LIBRARY_PATH), Cvoid, (UInt64,), seed)
end

"""
    solve_conjugate_gradient(matvec::Any, b::Any, x::Any, n::Csize_t, tolerance::Cdouble, max_iterations::Int32, user_data::Any) -> Any

Wrapper for C++ function: `solve_conjugate_gradient`

# Arguments
- `matvec::Ptr{Cvoid}` - Callback function
- `b::Ptr{Cdouble}`
- `x::Ptr{Cdouble}`
- `n::Csize_t`
- `tolerance::Cdouble`
- `max_iterations::Int32`
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
"""

function solve_conjugate_gradient(matvec::Any, b::Any, x::Any, n::Csize_t, tolerance::Cdouble, max_iterations::Int32, user_data::Any)::Status
    return ccall((:solve_conjugate_gradient, LIBRARY_PATH), Status, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Int32, Ptr{Cvoid},), matvec, b, x, n, tolerance, max_iterations, user_data)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_conjugate_gradient(matvec::Ptr{Cvoid}, b::Ptr{Cdouble}, x::Vector{Float64}, n::Csize_t, tolerance::Cdouble, max_iterations::Int32, user_data::Ptr{Cvoid})::Any
    return GC.@preserve x begin
        ccall((:solve_conjugate_gradient, LIBRARY_PATH), Any, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Int32, Ptr{Cvoid},), matvec, b, pointer(x), n, tolerance, max_iterations, user_data)
    end
end

"""
    solve_least_squares(A::Any, b::Any, x::Any) -> Any

Wrapper for C++ function: `solve_least_squares`

# Arguments
- `A::Ptr{DenseMatrix}`
- `b::Ptr{Cdouble}`
- `x::Ptr{Cdouble}`

# Returns
- `Any`

# Metadata
- Mangled symbol: `solve_least_squares`
"""

function solve_least_squares(A::Any, b::Any, x::Any)::Status
    return ccall((:solve_least_squares, LIBRARY_PATH), Status, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), A, b, x)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function solve_least_squares(A::DenseMatrix, b::Ptr{Cdouble}, x::Vector{Float64})::Any
    return GC.@preserve x begin
        ccall((:solve_least_squares, LIBRARY_PATH), Any, (Ptr{DenseMatrix}, Ptr{Cdouble}, Ptr{Cdouble},), Ref(A), b, pointer(x))
    end
end

"""
    solve_linear_system_lu(lu::Any, b::Any, x::Any, n::Csize_t) -> Any

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
"""

function solve_linear_system_lu(lu::Any, b::Any, x::Any, n::Csize_t)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_solve_linear_system_lu_thunk", lu, b, x, n)
end
"""
    solve_linear_system_qr(qr::Any, b::Any, x::Any) -> Any

Wrapper for C++ function: `solve_linear_system_qr`

# Arguments
- `qr::Ptr{QRDecomposition}`
- `b::Ptr{Cdouble}`
- `x::Ptr{Cdouble}`

# Returns
- `Any`

# Metadata
- Mangled symbol: `solve_linear_system_qr`
"""

function solve_linear_system_qr(qr::Any, b::Any, x::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_solve_linear_system_qr_thunk", qr, b, x)
end
"""
    solve_ode_adaptive(ode_func::Any, t0::Cdouble, t_final::Cdouble, y0::Any, n::Csize_t, tolerance::Cdouble, event_func::Any, user_data::Any) -> ODEResult

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
"""

function solve_ode_adaptive(ode_func::Any, t0::Cdouble, t_final::Cdouble, y0::Any, n::Csize_t, tolerance::Cdouble, event_func::Any, user_data::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_solve_ode_adaptive_thunk", ode_func, t0, t_final, y0, n, tolerance, event_func, user_data)
end
"""
    solve_ode_rk4(ode_func::Any, t0::Cdouble, t_final::Cdouble, y0::Any, n::Csize_t, dt::Cdouble, user_data::Any) -> ODEResult

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
"""

function solve_ode_rk4(ode_func::Any, t0::Cdouble, t_final::Cdouble, y0::Any, n::Csize_t, dt::Cdouble, user_data::Any)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_solve_ode_rk4_thunk", ode_func, t0, t_final, y0, n, dt, user_data)
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
"""

function sparse_matrix_create(rows::Csize_t, cols::Csize_t, nnz::Csize_t)
    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
    # Call the C-interface wrapper generated by llvm.emit_c_interface
    return RepliBuild.JITManager.invoke("_mlir_ciface_sparse_matrix_create_thunk", rows, cols, nnz)
end
"""
    sparse_matrix_destroy(mat::Any) -> Cvoid

Wrapper for C++ function: `sparse_matrix_destroy`

# Arguments
- `mat::Ptr{SparseMatrix}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `sparse_matrix_destroy`
"""

function sparse_matrix_destroy(mat::Any)::Cvoid
    ccall((:sparse_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{SparseMatrix},), mat)
end

# Convenience wrapper - accepts structs directly instead of pointers
function sparse_matrix_destroy(mat::SparseMatrix)::Cvoid
    return ccall((:sparse_matrix_destroy, LIBRARY_PATH), Cvoid, (Ptr{SparseMatrix},), Ref(mat))
end

"""
    spline_destroy(spline::Any) -> Cvoid

Wrapper for C++ function: `spline_destroy`

# Arguments
- `spline::Ptr{SplineInterpolation}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `spline_destroy`
"""

function spline_destroy(spline::Any)::Cvoid
    ccall((:spline_destroy, LIBRARY_PATH), Cvoid, (Ptr{SplineInterpolation},), spline)
end

"""
    spline_eval(spline::Any, x::Cdouble) -> Cdouble

Wrapper for C++ function: `spline_eval`

# Arguments
- `spline::Ptr{SplineInterpolation}`
- `x::Cdouble`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `spline_eval`
"""

function spline_eval(spline::Any, x::Cdouble)::Cdouble
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
"""

function status_to_string(status::Status)::String
    ptr = ccall((:status_to_string, LIBRARY_PATH), Cstring, (Status,), status)
    if ptr == C_NULL
        error("status_to_string returned NULL pointer")
    end
    return unsafe_string(ptr)
end

"""
    vector_axpy(y::Any, alpha::Cdouble, x::Any, n::Csize_t) -> Cvoid

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
"""

function vector_axpy(y::Any, alpha::Cdouble, x::Any, n::Csize_t)::Cvoid
    ccall((:vector_axpy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Ptr{Cdouble}, Csize_t,), y, alpha, x, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_axpy(y::Vector{Float64}, alpha::Cdouble, x::Vector{Float64}, n::Csize_t)::Cvoid
    return GC.@preserve y x begin
        ccall((:vector_axpy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Ptr{Cdouble}, Csize_t,), pointer(y), alpha, pointer(x), n)
    end
end

"""
    vector_copy(dest::Any, src::Any, n::Csize_t) -> Cvoid

Wrapper for C++ function: `vector_copy`

# Arguments
- `dest::Ptr{Cdouble}`
- `src::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `vector_copy`
"""

function vector_copy(dest::Any, src::Any, n::Csize_t)::Cvoid
    ccall((:vector_copy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), dest, src, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_copy(dest::Ptr{Cdouble}, src::Vector{Float64}, n::Csize_t)::Cvoid
    return GC.@preserve src begin
        ccall((:vector_copy, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), dest, pointer(src), n)
    end
end

"""
    vector_dot(x::Any, y::Any, n::Csize_t) -> Cdouble

Wrapper for C++ function: `vector_dot`

# Arguments
- `x::Ptr{Cdouble}`
- `y::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `vector_dot`
"""

function vector_dot(x::Any, y::Any, n::Csize_t)::Cdouble
    ccall((:vector_dot, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), x, y, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_dot(x::Vector{Float64}, y::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve x y begin
        ccall((:vector_dot, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), pointer(x), pointer(y), n)
    end
end

"""
    vector_norm(x::Any, n::Csize_t) -> Cdouble

Wrapper for C++ function: `vector_norm`

# Arguments
- `x::Ptr{Cdouble}`
- `n::Csize_t`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `vector_norm`
"""

function vector_norm(x::Any, n::Csize_t)::Cdouble
    ccall((:vector_norm, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), x, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_norm(x::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve x begin
        ccall((:vector_norm, LIBRARY_PATH), Cdouble, (Ptr{Cdouble}, Csize_t,), pointer(x), n)
    end
end

"""
    vector_scale(x::Any, alpha::Cdouble, n::Csize_t) -> Cvoid

Wrapper for C++ function: `vector_scale`

# Arguments
- `x::Ptr{Cdouble}`
- `alpha::Cdouble`
- `n::Csize_t`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `vector_scale`
"""

function vector_scale(x::Any, alpha::Cdouble, n::Csize_t)::Cvoid
    ccall((:vector_scale, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Csize_t,), x, alpha, n)
end

# Convenience wrapper - accepts arrays/structs directly with automatic GC preservation
function vector_scale(x::Vector{Float64}, alpha::Cdouble, n::Csize_t)::Cvoid
    return GC.@preserve x begin
        ccall((:vector_scale, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Cdouble, Csize_t,), pointer(x), alpha, n)
    end
end


end # module StressTest
