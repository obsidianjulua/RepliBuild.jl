# Auto-generated Julia wrapper for eigen_wrapper
# Generated: 2025-11-24 16:22:54
# Generator: RepliBuild Wrapper (Tier 3: Introspective)
# Library: libeigen_wrapper.so
# Metadata: compilation_metadata.json
#
# Type Safety: ✅ Perfect - Types extracted from compilation
# Language: Language-agnostic (via LLVM IR)
# Manual edits: None required

module EigenWrapper

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/examples/struct_test/examples/eigen_test/julia/libeigen_wrapper.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.5",
    "clang_version" => "clang version 21.1.5",
    "optimization" => "2",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 7,
    "generated_at" => "2025-11-24T16:22:49.491"
)

# =============================================================================
# Struct Definitions (from C++)
# =============================================================================

# C++ struct: Matrix3d (72 bytes)
mutable struct Matrix3d
    data::NTuple{9, Cdouble}  # 3x3 matrix in row-major order
end

# C++ struct: Vector3d (24 bytes)
mutable struct Vector3d
    x::Cdouble
    y::Cdouble
    z::Cdouble
end


export vec3_cross, vec3_create, mat3_mul_vec, mat3_identity, vec3_add, vec3_dot, vec3_norm

"""
    vec3_cross(arg1::Vector3d, arg2::Vector3d) -> Any

Wrapper for C++ function: `vec3_cross(Vector3d, Vector3d)`

# Arguments
- `arg1::Vector3d`
- `arg2::Vector3d`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z10vec3_cross8Vector3dS_`
- Type safety: ✅ From compilation
"""

function vec3_cross(arg1::Vector3d, arg2::Vector3d)::Vector3d
    return ccall((:_Z10vec3_cross8Vector3dS_, LIBRARY_PATH), Vector3d, (Vector3d, Vector3d,), arg1, arg2)
end

"""
    vec3_create(arg1::Cdouble, arg2::Cdouble, arg3::Cdouble) -> Any

Wrapper for C++ function: `vec3_create(double, double, double)`

# Arguments
- `arg1::Cdouble`
- `arg2::Cdouble`
- `arg3::Cdouble`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z11vec3_createddd`
- Type safety: ✅ From compilation
"""

function vec3_create(arg1::Cdouble, arg2::Cdouble, arg3::Cdouble)::Vector3d
    return ccall((:_Z11vec3_createddd, LIBRARY_PATH), Vector3d, (Cdouble, Cdouble, Cdouble,), arg1, arg2, arg3)
end

"""
    mat3_mul_vec(arg1::Matrix3d, arg2::Vector3d) -> Any

Wrapper for C++ function: `mat3_mul_vec(Matrix3d, Vector3d)`

# Arguments
- `arg1::Matrix3d`
- `arg2::Vector3d`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z12mat3_mul_vec8Matrix3d8Vector3d`
- Type safety: ✅ From compilation
"""

function mat3_mul_vec(arg1::Matrix3d, arg2::Vector3d)::Vector3d
    return ccall((:_Z12mat3_mul_vec8Matrix3d8Vector3d, LIBRARY_PATH), Vector3d, (Matrix3d, Vector3d,), arg1, arg2)
end

"""
    mat3_identity() -> Any

Wrapper for C++ function: `mat3_identity()`

# Arguments


# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z13mat3_identityv`
- Type safety: ✅ From compilation
"""

function mat3_identity()::Matrix3d
    return ccall((:_Z13mat3_identityv, LIBRARY_PATH), Matrix3d, (), )
end

"""
    vec3_add(arg1::Vector3d, arg2::Vector3d) -> Any

Wrapper for C++ function: `vec3_add(Vector3d, Vector3d)`

# Arguments
- `arg1::Vector3d`
- `arg2::Vector3d`

# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z8vec3_add8Vector3dS_`
- Type safety: ✅ From compilation
"""

function vec3_add(arg1::Vector3d, arg2::Vector3d)::Vector3d
    return ccall((:_Z8vec3_add8Vector3dS_, LIBRARY_PATH), Vector3d, (Vector3d, Vector3d,), arg1, arg2)
end

"""
    vec3_dot(arg1::Vector3d, arg2::Vector3d) -> Cdouble

Wrapper for C++ function: `vec3_dot(Vector3d, Vector3d)`

# Arguments
- `arg1::Vector3d`
- `arg2::Vector3d`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z8vec3_dot8Vector3dS_`
- Type safety: ✅ From compilation
"""

function vec3_dot(arg1::Vector3d, arg2::Vector3d)::Cdouble
    ccall((:_Z8vec3_dot8Vector3dS_, LIBRARY_PATH), Cdouble, (Vector3d, Vector3d,), arg1, arg2)
end

"""
    vec3_norm(arg1::Vector3d) -> Cdouble

Wrapper for C++ function: `vec3_norm(Vector3d)`

# Arguments
- `arg1::Vector3d`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z9vec3_norm8Vector3d`
- Type safety: ✅ From compilation
"""

function vec3_norm(arg1::Vector3d)::Cdouble
    ccall((:_Z9vec3_norm8Vector3d, LIBRARY_PATH), Cdouble, (Vector3d,), arg1)
end


end # module EigenWrapper
