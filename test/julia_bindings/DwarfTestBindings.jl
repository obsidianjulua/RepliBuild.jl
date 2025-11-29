# Auto-generated Julia wrapper for DwarfTest
# Generated: 2025-11-29 16:55:28
# Generator: RepliBuild Wrapper (Tier 3: Introspective)
# Library: libDwarfTest.so
# Metadata: compilation_metadata.json
#
# Type Safety:  Perfect - Types extracted from compilation
# Language: Language-agnostic (via LLVM IR)
# Manual edits: None required

module DwarfTestBindings

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/julia_bindings/libDwarfTest.so"

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
    "optimization" => "2",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 74,
    "generated_at" => "2025-11-29T16:55:25.502"
)

# =============================================================================
# Enum Definitions (from DWARF debug info)
# =============================================================================

# C++ enum: Color (underlying type: unsigned int)
@enum Color::Cuint begin
    RED = 0
    GREEN = 1
    BLUE = 2
end

# C++ enum: Priority (underlying type: unknown)
@enum Priority::Int32 begin
    LOW = -100
    MEDIUM = 0
    HIGH = 100
    CRITICAL = 1000
end

# C++ enum: Status (underlying type: unknown)
@enum Status::Int32 begin
    IDLE = 0
    RUNNING = 10
    STOPPED = 20
    ERROR = 255
end


# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: FixedArray<float, 10> (2 members)
mutable struct FixedArray_float_10
    data::NTuple{10, Cfloat}
    size::Cint
end

# C++ struct: Pair<int> (2 members)
mutable struct Pair_int
    first::Cint
    second::Cint
end

# C++ struct: Point2D (2 members)
mutable struct Point2D
    x::Cdouble
    y::Cdouble
end

# C++ struct: Shape (3 members)
mutable struct Shape
    _vptr_Shape::Ptr{Cvoid}
    color::Color
    id::Cint
end

# C++ struct: Vector3D (3 members)
mutable struct Vector3D
    x::Cdouble
    y::Cdouble
    z::Cdouble
end

# C++ struct: __va_list_tag (4 members)
mutable struct __va_list_tag
    gp_offset::Cuint
    fp_offset::Cuint
    overflow_arg_area::Ptr{Cvoid}
    reg_save_area::Ptr{Cvoid}
end

# C++ struct: BoundingBox (2 members)
mutable struct BoundingBox
    min::Point2D
    max::Point2D
end

# C++ struct: Circle (2 members)
mutable struct Circle
    radius::Cdouble
    center::Point2D
end

# C++ struct: Rectangle (3 members)
mutable struct Rectangle
    width::Cdouble
    height::Cdouble
    origin::Point2D
end


export abs, c_add, c_process, do_nothing, fill_array, get_status, get_string, count_chars, is_positive, log_message, blend_colors, create_array, modify_point, process_data, set_priority, create_circle, make_int_pair, compute_bounds, process_values, write_volatile, allocate_matrix, apply_binary_op, get_addition_op, get_const_point, get_volatile_ptr, is_high_priority, generate_sequence, get_global_vector, get_opaque_handle, register_callback, create_float_array, normalize_in_place, add, dot, cross, mul64, add_i8, combine, distance, get_char, get_name, midpoint, multiply, sum_ints, init_grid, math_deg_to_rad, math_pi, Shape_destroy_Shape, utils_clamp, Vector3D_destroy_Vector3D, Vector3D_operatorplusassign, Shape_getColor, Circle_area, Circle_getRadius, Circle_perimeter, Vector3D_length, Vector3D_normalize, Vector3D_operatorplus, Rectangle_area, Rectangle_perimeter, Color, RED, GREEN, BLUE, Status, IDLE, RUNNING, STOPPED, ERROR, Priority, LOW, MEDIUM, HIGH, CRITICAL, Point2D, Vector3D, Circle, Shape, FixedArray_float_10, __va_list_tag, Rectangle, BoundingBox, Pair_int

"""
    abs(x::Integer) -> Cint

Wrapper for C++ function: `abs`

# Arguments
- `x::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `abs`
- Type safety:  From compilation
"""

function abs(x::Integer)::Cint
    x_c = Cint(x)  # Auto-converts with overflow check
    return ccall((:abs, LIBRARY_PATH), Cint, (Cint,), x_c)
end

"""
    c_add(a::Integer, b::Integer) -> Cint

Wrapper for C++ function: `c_add`

# Arguments
- `a::Cint`
- `b::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `c_add`
- Type safety:  From compilation
"""

function c_add(a::Integer, b::Integer)::Cint
    a_c = Cint(a)  # Auto-converts with overflow check
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:c_add, LIBRARY_PATH), Cint, (Cint, Cint,), a_c, b_c)
end

"""
    c_process() -> Cvoid

Wrapper for C++ function: `c_process`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `c_process`
- Type safety:  From compilation
"""

function c_process()::Cvoid
    ccall((:c_process, LIBRARY_PATH), Cvoid, (), )
end

"""
    do_nothing() -> Cint

Wrapper for C++ function: `do_nothing()`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z10do_nothingv`
- Type safety:  From compilation
"""

function do_nothing()::Cint
    ccall((:_Z10do_nothingv, LIBRARY_PATH), Cint, (), )
end

"""
    fill_array(arg1::Ptr{Cvoid}, arg2::Integer, arg3::Integer) -> Cvoid

Wrapper for C++ function: `fill_array(int*, int, int)`

# Arguments
- `arg1::Ptr{Cvoid}`
- `arg2::Cint`
- `arg3::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z10fill_arrayPiii`
- Type safety:  From compilation
"""

function fill_array(arg1::Ptr{Cvoid}, arg2::Integer, arg3::Integer)::Cvoid
    arg2_c = Cint(arg2)  # Auto-converts with overflow check
    arg3_c = Cint(arg3)  # Auto-converts with overflow check
    return ccall((:_Z10fill_arrayPiii, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Cint, Cint,), arg1, arg2_c, arg3_c)
end

"""
    get_status() -> Any

Wrapper for C++ function: `get_status()`

# Arguments


# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z10get_statusv`
- Type safety:  From compilation
"""

function get_status()::Status
    return ccall((:_Z10get_statusv, LIBRARY_PATH), Status, (), )
end

"""
    get_string() -> Cstring

Wrapper for C++ function: `get_string()`

# Arguments


# Returns
- `Cstring`

# Metadata
- Mangled symbol: `_Z10get_stringv`
- Type safety:  From compilation
"""

function get_string()::String
    ptr = ccall((:_Z10get_stringv, LIBRARY_PATH), Cstring, (), )
    if ptr == C_NULL
        error("get_string returned NULL pointer")
    end
    return unsafe_string(ptr)
end

"""
    count_chars(str::Ptr{Cvoid}) -> Cint

Wrapper for C++ function: `count_chars(char const*)`

# Arguments
- `str::Ptr{Cvoid}`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z11count_charsPKc`
- Type safety:  From compilation
"""

function count_chars(str::Ptr{Cvoid})::Cint
    ccall((:_Z11count_charsPKc, LIBRARY_PATH), Cint, (Ptr{Cvoid},), str)
end

"""
    is_positive(x::Integer) -> Bool

Wrapper for C++ function: `is_positive(int)`

# Arguments
- `x::Cint`

# Returns
- `Bool`

# Metadata
- Mangled symbol: `_Z11is_positivei`
- Type safety:  From compilation
"""

function is_positive(x::Integer)::Bool
    x_c = Cint(x)  # Auto-converts with overflow check
    return ccall((:_Z11is_positivei, LIBRARY_PATH), Bool, (Cint,), x_c)
end

"""
    log_message(arg1::Ptr{Cvoid}, arg2::Any) -> Cvoid

Wrapper for C++ function: `log_message(char const*, ...)`

# Arguments
- `arg1::Ptr{Cvoid}`
- `arg2::Any`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z11log_messagePKcz`
- Type safety:  From compilation
"""

function log_message(arg1::Ptr{Cvoid}, arg2::Any)::Cvoid
    ccall((:_Z11log_messagePKcz, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Any,), arg1, arg2)
end

"""
    blend_colors(a::Color, b::Color) -> Any

Wrapper for C++ function: `blend_colors(Color, Color)`

# Arguments
- `a::Color`
- `b::Color`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z12blend_colors5ColorS_`
- Type safety:  From compilation
"""

function blend_colors(a::Color, b::Color)::Color
    return ccall((:_Z12blend_colors5ColorS_, LIBRARY_PATH), Color, (Color, Color,), a, b)
end

"""
    create_array(size::Integer) -> Ptr{Cvoid}

Wrapper for C++ function: `create_array(int)`

# Arguments
- `size::Cint`

# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `_Z12create_arrayi`
- Type safety:  From compilation
"""

function create_array(size::Integer)::Ptr{Cvoid}
    size_c = Cint(size)  # Auto-converts with overflow check
    return ccall((:_Z12create_arrayi, LIBRARY_PATH), Ptr{Cvoid}, (Cint,), size_c)
end

"""
    modify_point(arg1::Ptr{Point2D}) -> Cvoid

Wrapper for C++ function: `modify_point(Point2D*)`

# Arguments
- `arg1::Ptr{Point2D}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z12modify_pointP7Point2D`
- Type safety:  From compilation
"""

function modify_point(arg1::Ptr{Point2D})::Cvoid
    ccall((:_Z12modify_pointP7Point2D, LIBRARY_PATH), Cvoid, (Ptr{Point2D},), arg1)
end

"""
    process_data(arg1::Ptr{Cvoid}, arg2::Integer) -> Cvoid

Wrapper for C++ function: `process_data(void*, int)`

# Arguments
- `arg1::Ptr{Cvoid}`
- `arg2::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z12process_dataPvi`
- Type safety:  From compilation
"""

function process_data(arg1::Ptr{Cvoid}, arg2::Integer)::Cvoid
    arg2_c = Cint(arg2)  # Auto-converts with overflow check
    return ccall((:_Z12process_dataPvi, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Cint,), arg1, arg2_c)
end

"""
    set_priority(arg1::Priority) -> Cvoid

Wrapper for C++ function: `set_priority(Priority)`

# Arguments
- `arg1::Priority`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z12set_priority8Priority`
- Type safety:  From compilation
"""

function set_priority(arg1::Priority)::Cvoid
    ccall((:_Z12set_priority8Priority, LIBRARY_PATH), Cvoid, (Priority,), arg1)
end

"""
    create_circle(center::Point2D, radius::Cdouble, color::Color) -> Circle

Wrapper for C++ function: `create_circle(Point2D, double, Color)`

# Arguments
- `center::Point2D`
- `radius::Cdouble`
- `color::Color`

# Returns
- `Circle`

# Metadata
- Mangled symbol: `_Z13create_circle7Point2Dd5Color`
- Type safety:  From compilation
"""

function create_circle(center::Point2D, radius::Cdouble, color::Color)::Circle
    ccall((:_Z13create_circle7Point2Dd5Color, LIBRARY_PATH), Circle, (Point2D, Cdouble, Color,), center, radius, color)
end

"""
    make_int_pair(a::Integer, b::Integer) -> Any

Wrapper for C++ function: `make_int_pair(int, int)`

# Arguments
- `a::Cint`
- `b::Cint`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z13make_int_pairii`
- Type safety:  From compilation
"""

function make_int_pair(a::Integer, b::Integer)::Any
    a_c = Cint(a)  # Auto-converts with overflow check
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:_Z13make_int_pairii, LIBRARY_PATH), Any, (Cint, Cint,), a_c, b_c)
end

"""
    compute_bounds(points::Ptr{Point2D}, count::Integer) -> BoundingBox

Wrapper for C++ function: `compute_bounds(Point2D const*, int)`

# Arguments
- `points::Ptr{Point2D}`
- `count::Cint`

# Returns
- `BoundingBox`

# Metadata
- Mangled symbol: `_Z14compute_boundsPK7Point2Di`
- Type safety:  From compilation
"""

function compute_bounds(points::Ptr{Point2D}, count::Integer)::BoundingBox
    count_c = Cint(count)  # Auto-converts with overflow check
    return ccall((:_Z14compute_boundsPK7Point2Di, LIBRARY_PATH), BoundingBox, (Ptr{Point2D}, Cint,), points, count_c)
end

"""
    process_values(arg1::Ptr{Cvoid}, arg2::Integer) -> Cvoid

Wrapper for C++ function: `process_values(double const*, int)`

# Arguments
- `arg1::Ptr{Cvoid}`
- `arg2::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z14process_valuesPKdi`
- Type safety:  From compilation
"""

function process_values(arg1::Ptr{Cvoid}, arg2::Integer)::Cvoid
    arg2_c = Cint(arg2)  # Auto-converts with overflow check
    return ccall((:_Z14process_valuesPKdi, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Cint,), arg1, arg2_c)
end

"""
    write_volatile(arg1::Ptr{Cvoid}, arg2::Integer) -> Cvoid

Wrapper for C++ function: `write_volatile(int volatile*, int)`

# Arguments
- `arg1::Ptr{Cvoid}`
- `arg2::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z14write_volatilePVii`
- Type safety:  From compilation
"""

function write_volatile(arg1::Ptr{Cvoid}, arg2::Integer)::Cvoid
    arg2_c = Cint(arg2)  # Auto-converts with overflow check
    return ccall((:_Z14write_volatilePVii, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Cint,), arg1, arg2_c)
end

"""
    allocate_matrix(arg1::Ptr{Cvoid}, arg2::Integer, arg3::Integer) -> Cvoid

Wrapper for C++ function: `allocate_matrix(double**, int, int)`

# Arguments
- `arg1::Ptr{Cvoid}`
- `arg2::Cint`
- `arg3::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z15allocate_matrixPPdii`
- Type safety:  From compilation
"""

function allocate_matrix(arg1::Ptr{Cvoid}, arg2::Integer, arg3::Integer)::Cvoid
    arg2_c = Cint(arg2)  # Auto-converts with overflow check
    arg3_c = Cint(arg3)  # Auto-converts with overflow check
    return ccall((:_Z15allocate_matrixPPdii, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Cint, Cint,), arg1, arg2_c, arg3_c)
end

"""
    apply_binary_op(a::Integer, b::Integer, op::Any) -> Cint

Wrapper for C++ function: `apply_binary_op(int, int, int (*)(int, int))`

# Arguments
- `a::Cint`
- `b::Cint`
- `op::Any`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z15apply_binary_opiiPFiiiE`
- Type safety:  From compilation
"""

function apply_binary_op(a::Integer, b::Integer, op::Any)::Cint
    a_c = Cint(a)  # Auto-converts with overflow check
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:_Z15apply_binary_opiiPFiiiE, LIBRARY_PATH), Cint, (Cint, Cint, Any,), a_c, b_c, op)
end

"""
    get_addition_op() -> Any

Wrapper for C++ function: `get_addition_op()`

# Arguments


# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z15get_addition_opv`
- Type safety:  From compilation
"""

function get_addition_op()::Any
    ccall((:_Z15get_addition_opv, LIBRARY_PATH), Any, (), )
end

"""
    get_const_point() -> Ptr{Cvoid}

Wrapper for C++ function: `get_const_point()`

# Arguments


# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `_Z15get_const_pointv`
- Type safety:  From compilation
"""

function get_const_point()::Ptr{Cvoid}
    ccall((:_Z15get_const_pointv, LIBRARY_PATH), Ptr{Cvoid}, (), )
end

"""
    get_volatile_ptr() -> Ptr{Cvoid}

Wrapper for C++ function: `get_volatile_ptr()`

# Arguments


# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `_Z16get_volatile_ptrv`
- Type safety:  From compilation
"""

function get_volatile_ptr()::Ptr{Cvoid}
    ccall((:_Z16get_volatile_ptrv, LIBRARY_PATH), Ptr{Cvoid}, (), )
end

"""
    is_high_priority(p::Priority) -> Bool

Wrapper for C++ function: `is_high_priority(Priority)`

# Arguments
- `p::Priority`

# Returns
- `Bool`

# Metadata
- Mangled symbol: `_Z16is_high_priority8Priority`
- Type safety:  From compilation
"""

function is_high_priority(p::Priority)::Bool
    ccall((:_Z16is_high_priority8Priority, LIBRARY_PATH), Bool, (Priority,), p)
end

"""
    generate_sequence(start::Integer, count::Integer) -> Ptr{Cvoid}

Wrapper for C++ function: `generate_sequence(int, int)`

# Arguments
- `start::Cint`
- `count::Cint`

# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `_Z17generate_sequenceii`
- Type safety:  From compilation
"""

function generate_sequence(start::Integer, count::Integer)::Ptr{Cvoid}
    start_c = Cint(start)  # Auto-converts with overflow check
    count_c = Cint(count)  # Auto-converts with overflow check
    return ccall((:_Z17generate_sequenceii, LIBRARY_PATH), Ptr{Cvoid}, (Cint, Cint,), start_c, count_c)
end

"""
    get_global_vector() -> Ref{Cvoid}

Wrapper for C++ function: `get_global_vector()`

# Arguments


# Returns
- `Ref{Cvoid}`

# Metadata
- Mangled symbol: `_Z17get_global_vectorv`
- Type safety:  From compilation
"""

function get_global_vector()::Ref{Cvoid}
    ccall((:_Z17get_global_vectorv, LIBRARY_PATH), Ref{Cvoid}, (), )
end

"""
    get_opaque_handle() -> Ptr{Cvoid}

Wrapper for C++ function: `get_opaque_handle()`

# Arguments


# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `_Z17get_opaque_handlev`
- Type safety:  From compilation
"""

function get_opaque_handle()::Ptr{Cvoid}
    ccall((:_Z17get_opaque_handlev, LIBRARY_PATH), Ptr{Cvoid}, (), )
end

"""
    register_callback(arg1::Ptr{Cvoid}) -> Cvoid

Wrapper for C++ function: `register_callback(void (*)(void*, char const*), void*)`

# Arguments
- `arg1::Ptr{Cvoid}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z17register_callbackPFvPvPKcES_`
- Type safety:  From compilation
"""

function register_callback(arg1::Ptr{Cvoid})::Cvoid
    ccall((:_Z17register_callbackPFvPvPKcES_, LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), arg1)
end

"""
    create_float_array() -> Any

Wrapper for C++ function: `create_float_array()`

# Arguments


# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z18create_float_arrayv`
- Type safety:  From compilation
"""

function create_float_array()::Any
    ccall((:_Z18create_float_arrayv, LIBRARY_PATH), Any, (), )
end

"""
    normalize_in_place(arg1::Ref{Vector3D}) -> Cvoid

Wrapper for C++ function: `normalize_in_place(Vector3D&)`

# Arguments
- `arg1::Ref{Vector3D}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z18normalize_in_placeR8Vector3D`
- Type safety:  From compilation
"""

function normalize_in_place(arg1::Ref{Vector3D})::Cvoid
    ccall((:_Z18normalize_in_placeR8Vector3D, LIBRARY_PATH), Cvoid, (Ref{Vector3D},), arg1)
end

"""
    abs(x::Cdouble) -> Cdouble

Wrapper for C++ function: `abs(double)`

# Arguments
- `x::Cdouble`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z3absd`
- Type safety:  From compilation
"""

function abs(x::Cdouble)::Cdouble
    ccall((:_Z3absd, LIBRARY_PATH), Cdouble, (Cdouble,), x)
end

"""
    abs(x::Cfloat) -> Cfloat

Wrapper for C++ function: `abs(float)`

# Arguments
- `x::Cfloat`

# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `_Z3absf`
- Type safety:  From compilation
"""

function abs(x::Cfloat)::Cfloat
    ccall((:_Z3absf, LIBRARY_PATH), Cfloat, (Cfloat,), x)
end

"""
    add(a::Integer, b::Integer) -> Cint

Wrapper for C++ function: `add(int, int)`

# Arguments
- `a::Cint`
- `b::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z3addii`
- Type safety:  From compilation
"""

function add(a::Integer, b::Integer)::Cint
    a_c = Cint(a)  # Auto-converts with overflow check
    b_c = Cint(b)  # Auto-converts with overflow check
    return ccall((:_Z3addii, LIBRARY_PATH), Cint, (Cint, Cint,), a_c, b_c)
end

"""
    dot(a::Ref{Vector3D}, b::Ref{Vector3D}) -> Cdouble

Wrapper for C++ function: `dot(Vector3D const&, Vector3D const&)`

# Arguments
- `a::Ref{Vector3D}`
- `b::Ref{Vector3D}`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z3dotRK8Vector3DS1_`
- Type safety:  From compilation
"""

function dot(a::Ref{Vector3D}, b::Ref{Vector3D})::Cdouble
    ccall((:_Z3dotRK8Vector3DS1_, LIBRARY_PATH), Cdouble, (Ref{Vector3D}, Ref{Vector3D},), a, b)
end

"""
    cross(a::Vector3D, b::Vector3D) -> Vector3D

Wrapper for C++ function: `cross(Vector3D, Vector3D)`

# Arguments
- `a::Vector3D`
- `b::Vector3D`

# Returns
- `Vector3D`

# Metadata
- Mangled symbol: `_Z5cross8Vector3DS_`
- Type safety:  From compilation
"""

function cross(a::Vector3D, b::Vector3D)::Vector3D
    ccall((:_Z5cross8Vector3DS_, LIBRARY_PATH), Vector3D, (Vector3D, Vector3D,), a, b)
end

"""
    mul64(a::Any, b::Any) -> Any

Wrapper for C++ function: `mul64(int, int)`

# Arguments
- `a::Any`
- `b::Any`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z5mul64ii`
- Type safety:  From compilation
"""

function mul64(a::Any, b::Any)::Any
    ccall((:_Z5mul64ii, LIBRARY_PATH), Any, (Any, Any,), a, b)
end

"""
    add_i8(a::Any, b::Any) -> Any

Wrapper for C++ function: `add_i8(signed char, signed char)`

# Arguments
- `a::Any`
- `b::Any`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z6add_i8aa`
- Type safety:  From compilation
"""

function add_i8(a::Any, b::Any)::Any
    ccall((:_Z6add_i8aa, LIBRARY_PATH), Any, (Any, Any,), a, b)
end

"""
    combine(hi::Any, lo::Any) -> Any

Wrapper for C++ function: `combine(unsigned short, unsigned short)`

# Arguments
- `hi::Any`
- `lo::Any`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z7combinett`
- Type safety:  From compilation
"""

function combine(hi::Any, lo::Any)::Any
    ccall((:_Z7combinett, LIBRARY_PATH), Any, (Any, Any,), hi, lo)
end

"""
    distance(a::Point2D, b::Point2D) -> Cdouble

Wrapper for C++ function: `distance(Point2D, Point2D)`

# Arguments
- `a::Point2D`
- `b::Point2D`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z8distance7Point2DS_`
- Type safety:  From compilation
"""

function distance(a::Point2D, b::Point2D)::Cdouble
    ccall((:_Z8distance7Point2DS_, LIBRARY_PATH), Cdouble, (Point2D, Point2D,), a, b)
end

"""
    get_char(str::Ptr{Cvoid}, index::Integer) -> Cchar

Wrapper for C++ function: `get_char(char const*, int)`

# Arguments
- `str::Ptr{Cvoid}`
- `index::Cint`

# Returns
- `Cchar`

# Metadata
- Mangled symbol: `_Z8get_charPKci`
- Type safety:  From compilation
"""

function get_char(str::Ptr{Cvoid}, index::Integer)::Cchar
    index_c = Cint(index)  # Auto-converts with overflow check
    return ccall((:_Z8get_charPKci, LIBRARY_PATH), Cchar, (Ptr{Cvoid}, Cint,), str, index_c)
end

"""
    get_name() -> Cstring

Wrapper for C++ function: `get_name()`

# Arguments


# Returns
- `Cstring`

# Metadata
- Mangled symbol: `_Z8get_namev`
- Type safety:  From compilation
"""

function get_name()::String
    ptr = ccall((:_Z8get_namev, LIBRARY_PATH), Cstring, (), )
    if ptr == C_NULL
        error("get_name returned NULL pointer")
    end
    return unsafe_string(ptr)
end

"""
    midpoint(a::Point2D, b::Point2D) -> Point2D

Wrapper for C++ function: `midpoint(Point2D, Point2D)`

# Arguments
- `a::Point2D`
- `b::Point2D`

# Returns
- `Point2D`

# Metadata
- Mangled symbol: `_Z8midpoint7Point2DS_`
- Type safety:  From compilation
"""

function midpoint(a::Point2D, b::Point2D)::Point2D
    ccall((:_Z8midpoint7Point2DS_, LIBRARY_PATH), Point2D, (Point2D, Point2D,), a, b)
end

"""
    multiply(x::Cdouble, y::Cdouble) -> Cdouble

Wrapper for C++ function: `multiply(double, double)`

# Arguments
- `x::Cdouble`
- `y::Cdouble`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z8multiplydd`
- Type safety:  From compilation
"""

function multiply(x::Cdouble, y::Cdouble)::Cdouble
    ccall((:_Z8multiplydd, LIBRARY_PATH), Cdouble, (Cdouble, Cdouble,), x, y)
end

"""
    sum_ints(count::Integer) -> Cint

Wrapper for C++ function: `sum_ints(int, ...)`

# Arguments
- `count::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z8sum_intsiz`
- Type safety:  From compilation
"""

function sum_ints(count::Integer)::Cint
    count_c = Cint(count)  # Auto-converts with overflow check
    return ccall((:_Z8sum_intsiz, LIBRARY_PATH), Cint, (Cint,), count_c)
end

"""
    init_grid(arg1::Ptr{Cvoid}, arg2::Integer, arg3::Integer) -> Cvoid

Wrapper for C++ function: `init_grid(Grid*, int, int)`

# Arguments
- `arg1::Ptr{Cvoid}`
- `arg2::Cint`
- `arg3::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_Z9init_gridP4Gridii`
- Type safety:  From compilation
"""

function init_grid(arg1::Ptr{Cvoid}, arg2::Integer, arg3::Integer)::Cvoid
    arg2_c = Cint(arg2)  # Auto-converts with overflow check
    arg3_c = Cint(arg3)  # Auto-converts with overflow check
    return ccall((:_Z9init_gridP4Gridii, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Cint, Cint,), arg1, arg2_c, arg3_c)
end

"""
    math_deg_to_rad(deg::Cdouble) -> Cdouble

Wrapper for C++ function: `math::deg_to_rad(double)`

# Arguments
- `deg::Cdouble`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZN4math10deg_to_radEd`
- Type safety:  From compilation
"""

function math_deg_to_rad(deg::Cdouble)::Cdouble
    ccall((:_ZN4math10deg_to_radEd, LIBRARY_PATH), Cdouble, (Cdouble,), deg)
end

"""
    math_pi() -> Cdouble

Wrapper for C++ function: `math::pi()`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZN4math2piEv`
- Type safety:  From compilation
"""

function math_pi()::Cdouble
    ccall((:_ZN4math2piEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Shape_destroy_Shape() -> Cvoid

Wrapper for C++ function: `Shape::~Shape()`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_ZN5ShapeD0Ev`
- Type safety:  From compilation
"""

function Shape_destroy_Shape()::Cvoid
    ccall((:_ZN5ShapeD0Ev, LIBRARY_PATH), Cvoid, (), )
end

"""
    Shape_destroy_Shape() -> Cvoid

Wrapper for C++ function: `Shape::~Shape()`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_ZN5ShapeD2Ev`
- Type safety:  From compilation
"""

function Shape_destroy_Shape()::Cvoid
    ccall((:_ZN5ShapeD2Ev, LIBRARY_PATH), Cvoid, (), )
end

"""
    Shape_destroy_Shape() -> Cvoid

Wrapper for C++ function: `Shape::~Shape()`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_ZN5ShapeD2Ev`
- Type safety:  From compilation
"""

function Shape_destroy_Shape()::Cvoid
    ccall((:_ZN5ShapeD2Ev, LIBRARY_PATH), Cvoid, (), )
end

"""
    utils_clamp(value::Integer, min::Integer, max::Integer) -> Cint

Wrapper for C++ function: `utils::clamp(int, int, int)`

# Arguments
- `value::Cint`
- `min::Cint`
- `max::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `_ZN5utils5clampEiii`
- Type safety:  From compilation
"""

function utils_clamp(value::Integer, min::Integer, max::Integer)::Cint
    value_c = Cint(value)  # Auto-converts with overflow check
    min_c = Cint(min)  # Auto-converts with overflow check
    max_c = Cint(max)  # Auto-converts with overflow check
    return ccall((:_ZN5utils5clampEiii, LIBRARY_PATH), Cint, (Cint, Cint, Cint,), value_c, min_c, max_c)
end

"""
    Vector3D_destroy_Vector3D() -> Cvoid

Wrapper for C++ function: `Vector3D::~Vector3D()`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_ZN8Vector3DD2Ev`
- Type safety:  From compilation
"""

function Vector3D_destroy_Vector3D()::Cvoid
    ccall((:_ZN8Vector3DD2Ev, LIBRARY_PATH), Cvoid, (), )
end

"""
    Vector3D_destroy_Vector3D() -> Cvoid

Wrapper for C++ function: `Vector3D::~Vector3D()`

# Arguments


# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `_ZN8Vector3DD2Ev`
- Type safety:  From compilation
"""

function Vector3D_destroy_Vector3D()::Cvoid
    ccall((:_ZN8Vector3DD2Ev, LIBRARY_PATH), Cvoid, (), )
end

"""
    Vector3D_operatorplusassign(arg1::Ref{Any}) -> Ref{Cvoid}

Wrapper for C++ function: `Vector3D::operator+=(Vector3D const&)`

# Arguments
- `arg1::Ref{Any}`

# Returns
- `Ref{Cvoid}`

# Metadata
- Mangled symbol: `_ZN8Vector3DpLERKS_`
- Type safety:  From compilation
"""

function Vector3D_operatorplusassign(arg1::Ref{Any})::Ref{Cvoid}
    ccall((:_ZN8Vector3DpLERKS_, LIBRARY_PATH), Ref{Cvoid}, (Ref{Any},), arg1)
end

"""
    Shape_getColor() -> Any

Wrapper for C++ function: `Shape::getColor() const`

# Arguments


# Returns
- `Any`

# Metadata
- Mangled symbol: `_ZNK5Shape8getColorEv`
- Type safety:  From compilation
"""

function Shape_getColor()::Color
    return ccall((:_ZNK5Shape8getColorEv, LIBRARY_PATH), Color, (), )
end

"""
    Circle_area() -> Cdouble

Wrapper for C++ function: `Circle::area() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle4areaEv`
- Type safety:  From compilation
"""

function Circle_area()::Cdouble
    ccall((:_ZNK6Circle4areaEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Circle_getRadius() -> Cdouble

Wrapper for C++ function: `Circle::getRadius() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle9getRadiusEv`
- Type safety:  From compilation
"""

function Circle_getRadius()::Cdouble
    ccall((:_ZNK6Circle9getRadiusEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Circle_perimeter() -> Cdouble

Wrapper for C++ function: `Circle::perimeter() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK6Circle9perimeterEv`
- Type safety:  From compilation
"""

function Circle_perimeter()::Cdouble
    ccall((:_ZNK6Circle9perimeterEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Vector3D_length() -> Cdouble

Wrapper for C++ function: `Vector3D::length() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK8Vector3D6lengthEv`
- Type safety:  From compilation
"""

function Vector3D_length()::Cdouble
    ccall((:_ZNK8Vector3D6lengthEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Vector3D_normalize() -> Vector3D

Wrapper for C++ function: `Vector3D::normalize() const`

# Arguments


# Returns
- `Vector3D`

# Metadata
- Mangled symbol: `_ZNK8Vector3D9normalizeEv`
- Type safety:  From compilation
"""

function Vector3D_normalize()::Vector3D
    ccall((:_ZNK8Vector3D9normalizeEv, LIBRARY_PATH), Vector3D, (), )
end

"""
    Vector3D_operatorplus(arg1::Ref{Any}) -> Vector3D

Wrapper for C++ function: `Vector3D::operator+(Vector3D const&) const`

# Arguments
- `arg1::Ref{Any}`

# Returns
- `Vector3D`

# Metadata
- Mangled symbol: `_ZNK8Vector3DplERKS_`
- Type safety:  From compilation
"""

function Vector3D_operatorplus(arg1::Ref{Any})::Vector3D
    ccall((:_ZNK8Vector3DplERKS_, LIBRARY_PATH), Vector3D, (Ref{Any},), arg1)
end

"""
    Rectangle_area() -> Cdouble

Wrapper for C++ function: `Rectangle::area() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle4areaEv`
- Type safety:  From compilation
"""

function Rectangle_area()::Cdouble
    ccall((:_ZNK9Rectangle4areaEv, LIBRARY_PATH), Cdouble, (), )
end

"""
    Rectangle_perimeter() -> Cdouble

Wrapper for C++ function: `Rectangle::perimeter() const`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_ZNK9Rectangle9perimeterEv`
- Type safety:  From compilation
"""

function Rectangle_perimeter()::Cdouble
    ccall((:_ZNK9Rectangle9perimeterEv, LIBRARY_PATH), Cdouble, (), )
end


end # module DwarfTestBindings
