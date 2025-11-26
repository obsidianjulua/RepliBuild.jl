# Auto-generated Julia wrapper for AdvancedTypes
# Generated: 2025-11-26 18:30:41
# Generator: RepliBuild Wrapper (Tier 1: Basic)
# Library: libAdvancedTypes.so
#
#   TYPE SAFETY: BASIC (40%)
# This wrapper uses conservative type placeholders extracted from binary symbols.
# For production use, regenerate with headers: RepliBuild.wrap(lib, headers=["mylib.h"])

module AdvancedTypes

using Libdl

# =============================================================================
# LIBRARY MANAGEMENT
# =============================================================================

const _LIB_PATH = raw"/home/grim/Desktop/Projects/RepliBuild.jl/test/build_advanced/julia/libAdvancedTypes.so"
const _LIB = Ref{Ptr{Nothing}}(C_NULL)
const _LOAD_ERRORS = String[]

function __init__()
    try
        _LIB[] = Libdl.dlopen(_LIB_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    catch e
        push!(_LOAD_ERRORS, string(e))
        @error "Failed to load library libAdvancedTypes.so" exception=e
    end
end

"""
    is_loaded()

Check if the library is successfully loaded.
"""
is_loaded() = _LIB[] != C_NULL

"""
    get_load_errors()

Get any errors that occurred during library loading.
"""
get_load_errors() = copy(_LOAD_ERRORS)

"""
    get_lib_path()

Get the path to the underlying library.
"""
get_lib_path() = _LIB_PATH

# Safety check macro
macro check_loaded()
    quote
        if !is_loaded()
            error("Library not loaded. Errors: ", join(get_load_errors(), "; "))
        end
    end
end

# =============================================================================
# FUNCTION WRAPPERS
# =============================================================================

"""
    matrix_sum_Matrix3x3_(args...)

Wrapper for C/C++ function `matrix_sum(Matrix3x3)`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`matrix_sum(Matrix3x3)`
"""
function matrix_sum_Matrix3x3_(args...)
    @check_loaded()
    ccall((:matrix_sum(Matrix3x3), _LIB[]), Any, (), args...)
end


"""
    add_callback_double_double_(args...)

Wrapper for C/C++ function `add_callback(double, double)`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`add_callback(double, double)`
"""
function add_callback_double_double_(args...)
    @check_loaded()
    ccall((:add_callback(double, double), _LIB[]), Any, (), args...)
end


"""
    check_status_Status_(args...)

Wrapper for C/C++ function `check_status(Status)`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`check_status(Status)`
"""
function check_status_Status_(args...)
    @check_loaded()
    ccall((:check_status(Status), _LIB[]), Any, (), args...)
end


"""
    color_to_int_Color_(args...)

Wrapper for C/C++ function `color_to_int(Color)`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`color_to_int(Color)`
"""
function color_to_int_Color_(args...)
    @check_loaded()
    ccall((:color_to_int(Color), _LIB[]), Any, (), args...)
end


"""
    apply_callback_int_double_double_double_double_(args...)

Wrapper for C/C++ function `apply_callback(int (*)(double, double), double, double)`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`apply_callback(int (*)(double, double), double, double)`
"""
function apply_callback_int_double_double_double_double_(args...)
    @check_loaded()
    ccall((:apply_callback(int (*)(double, double), double, double), _LIB[]), Any, (), args...)
end


"""
    create_complex_Color_Status_int_double_double_(args...)

Wrapper for C/C++ function `create_complex(Color, Status, int (*)(double, double))`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`create_complex(Color, Status, int (*)(double, double))`
"""
function create_complex_Color_Status_int_double_double_(args...)
    @check_loaded()
    ccall((:create_complex(Color, Status, int (*)(double, double)), _LIB[]), Any, (), args...)
end


"""
    get_primary_color_(args...)

Wrapper for C/C++ function `get_primary_color()`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`get_primary_color()`
"""
function get_primary_color_(args...)
    @check_loaded()
    ccall((:get_primary_color(), _LIB[]), Any, (), args...)
end


"""
    create_identity_matrix_(args...)

Wrapper for C/C++ function `create_identity_matrix()`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`create_identity_matrix()`
"""
function create_identity_matrix_(args...)
    @check_loaded()
    ccall((:create_identity_matrix(), _LIB[]), Any, (), args...)
end


"""
    grid_get_Grid_int_int_(args...)

Wrapper for C/C++ function `grid_get(Grid, int, int)`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`grid_get(Grid, int, int)`
"""
function grid_get_Grid_int_int_(args...)
    @check_loaded()
    ccall((:grid_get(Grid, int, int), _LIB[]), Any, (), args...)
end


"""
    run_tests_(args...)

Wrapper for C/C++ function `run_tests()`.

# Type Safety:   BASIC
Signature uses placeholder types. Actual types unknown without headers.
Return type and parameters may need manual adjustment.

# C/C++ Symbol
`run_tests()`
"""
function run_tests_(args...)
    @check_loaded()
    ccall((:run_tests(), _LIB[]), Any, (), args...)
end


# =============================================================================
# METADATA
# =============================================================================

"""
    library_info()

Get information about the wrapped library.
"""
function library_info()
    return Dict{Symbol,Any}(
        :name => "AdvancedTypes",
        :path => _LIB_PATH,
        :loaded => is_loaded(),
        :tier => :basic,
        :type_safety => "40% (conservative placeholders)",
        :functions_wrapped => 10,
        :functions_total => 10,
        :data_symbols => 0
    )
end

# Exports
export is_loaded, get_load_errors, get_lib_path, matrix_sum_Matrix3x3_, add_callback_double_double_, check_status_Status_, color_to_int_Color_, apply_callback_int_double_double_double_double_, create_complex_Color_Status_int_double_double_, get_primary_color_, create_identity_matrix_, grid_get_Grid_int_int_, run_tests_, library_info

end # module AdvancedTypes
