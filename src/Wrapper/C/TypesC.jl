# =============================================================================
# TYPE SYSTEM - Comprehensive C/C++ to Julia Type Mapping
# =============================================================================

# =============================================================================
# TYPE HEURISTICS - Smart detection of type categories
# =============================================================================

"""
    is_c_struct_like(c_type::String)::Bool

Check if a C++ type looks like a struct/class based on naming conventions.
Structs typically: start with uppercase, contain only alphanumeric + underscore.
"""
function is_c_struct_like(c_type::String)::Bool
    # Remove pointers, references, and whitespace
    base = strip(replace(replace(c_type, "*" => ""), "&" => ""))

    # Empty or primitive types are not structs
    if isempty(base) || base == "void"
        return false
    end

    # Struct names typically match: UpperCamelCase or snake_case with uppercase first letter
    # Examples: Matrix3x3, Grid, ComplexType, My_Struct
    return occursin(r"^[A-Z][A-Za-z0-9_]*$", base)
end

"""
    is_c_enum_like(c_type::String)::Bool

Check if a C type looks like an enum based on naming conventions.
Returns false — enum detection is unreliable by name alone (uppercase names
are more likely structs). Real enum identification uses DWARF metadata
via `_is_enum_type()` in DispatchLogic.jl which checks `__enum__` prefixed keys.
"""
function is_c_enum_like(c_type::String)::Bool
    # Enum detection requires DWARF metadata (__enum__ prefix).
    # Name-based heuristics can't distinguish enums from structs,
    # and treating an unknown struct as Cint would corrupt memory.
    return false
end

"""
    is_c_function_pointer_like(c_type::String)::Bool

Check if a C++ type contains function pointer syntax.
Looks for patterns: (*name), (*)(args), (^name) (blocks)
"""
function is_c_function_pointer_like(c_type::String)::Bool
    # Function pointer patterns:
    # - int (*callback)(double, double)
    # - void (*cleanup)()
    # - typedef int (*IntCallback)(double, double)
    return occursin(r"\(\s*\*", c_type) || occursin(r"\(\s*\^", c_type)
end





# =============================================================================
# A SCALAR THE TYPE MAP DOES NOT KNOW, AND 128-BIT INTEGERS
# =============================================================================

"""
    _unmapped_crossings(params, param_types) -> Vector{String}

The parameters a function takes BY VALUE that the type map has no Julia type
for, one phrase each naming the C type; empty when every one is mapped.

`params` are the function's parameters and `param_types` their `ccall` types in
the same order (`_ccall_param_type`): a parameter still `Any` there is a C
scalar the map does not know (`_Float16`, `__float128`, a spelling a compiler
added since). A return the map does not know is said the same way
(`_unmapped_return`) by the path that would have emitted it.

`Any` is never a foreign-call type, and the C generator used to write it
anyway: a parameter as `(Any,)`, which `_assert_no_any_ccall_argument` refuses
by refusing the whole wrapper, and a return by its C spelling, which for
`unsigned __int128` is not even Julia syntax. One such function then cost the
library every other one. Each is a trap stub instead (`_unmapped_trap_body`).
`long double` has its own (`_x87_crossings`), asked first.
"""
function _unmapped_crossings(params, param_types)::Vector{String}
    out = String[]
    for (p, t) in zip(params, param_types)
        t == "Any" || continue
        push!(out, "takes `$(get(p, "c_type", "?"))` $(get(p, "name", "")) by value")
    end
    return out
end

"The phrase for a return the type map does not know; see `_unmapped_crossings`."
_unmapped_return(c_type::AbstractString) = "returns `$c_type`"

"The body of the trap `_unmapped_crossings` calls for."
_unmapped_trap_body(julia_name::AbstractString, crossings::Vector{String}) =
    "    Base.error(" * repr(
        "ABI Safety Trap: cannot call '$julia_name'. It $(join(crossings, " and ")). " *
        "RepliBuild maps that C type to no Julia type, so no ccall could pass or return it. " *
        "Keep the function out with [wrap] exclude_symbols, or wrap a shim that takes and " *
        "returns the value through a pointer.") * ")\n"

const _C_INT128_TYPES = ("Int128", "UInt128")

# The `ccall` types that take no general-purpose register on x86-64 SysV.
const _C_SSE_SCALARS = ("Cfloat", "Cdouble", "Float32", "Float64", "ComplexF32", "ComplexF64")
# ...and the ones that take exactly one: every integer of 8 bytes or less.
const _C_GPR_SCALARS = ("Bool", "Cchar", "Cuchar", "Int8", "UInt8", "Cshort", "Cushort", "Int16", "UInt16",
                        "Cint", "Cuint", "Int32", "UInt32", "Cwchar_t", "Clong", "Culong", "Clonglong",
                        "Culonglong", "Int64", "UInt64", "Csize_t", "Cssize_t", "Cptrdiff_t", "Cintptr_t",
                        "Cuintptr_t", "Ctime_t", "Cstring", "Cwstring")

"""
    _c_gpr_upper(ccall_type) -> Int

The most general-purpose registers an x86-64 SysV argument of this `ccall`
type takes: 0 for a float or `_Complex`, 1 for a pointer or an integer of 8
bytes or less, and 2 for anything else (a 128-bit integer; a record, which a
caller that knows its class can count exactly). An upper bound is what
`_c_int128_misplaced` needs: it only ever refuses more.
"""
_c_gpr_upper(t::AbstractString)::Int =
    t in _C_SSE_SCALARS ? 0 :
    (t in _C_GPR_SCALARS || startswith(t, "Ptr{") || startswith(t, "Ref{")) ? 1 : 2

"""
    _c_int128_misplaced(param_types, gpr_words, return_type, return_in_memory; abi) -> Vector{String}

The by-value 128-bit integers (`__int128`, `unsigned __int128`: `Int128` and
`UInt128` among the `ccall` types `param_types`, or as `return_type`) that
`ccall` would not pass or return as C does, one phrase each; empty when the
call is right. `gpr_words[i]` is the most general-purpose registers parameter
`i` can take (0 for a float, 1 for an integer or pointer, up to 2 for a small
record), and `return_in_memory` whether the return may take the hidden-pointer
register.

  * **x86-64 SysV.** C passes a 128-bit integer whole: in two general
    registers, or, when fewer than two are left, on the stack at a 16-byte
    boundary, leaving the last register for a later argument. Julia lowers
    `Int128` to `{ i64, i64 }` and LLVM places each half by itself: with one
    register left the low half takes it and the high half goes to the stack,
    and on the stack the pair sits at the next 8-byte slot. So the call is
    right exactly while both halves get registers, which is asked of an upper
    bound: the registers the return's hidden pointer and every parameter
    before it can take, plus two, must not exceed six. Measured against clang
    (`test_type_mapping_live.jl`): five `long`s and then an `unsigned
    __int128` read a wrong value, and so did seven. The return (RAX:RDX on
    both sides) is always right.
  * **Win64.** C takes one by reference, as `ccall` passes it, and returns it
    in XMM0 (clang and GCC for MinGW: `<2 x i64>`), where `ccall` expects a
    hidden pointer: a return is refused, a parameter is not.
  * **AAPCS64.** Julia passes `i128` itself, as clang does, and LLVM places it
    (an even register pair, or the stack): nothing is refused.

The C generator, which has no thunk, emits a trap for each. The C++ generator
sends a function crossing one to its thunk (`is_ccall_safe`), which passes
`i128`: LLVM places that as clang's callee reads it on SysV and AAPCS64, and
`FunctionGen.thunk_crossings` refuses it under Win64.
"""
function _c_int128_misplaced(param_types, gpr_words, return_type::AbstractString,
                             return_in_memory::Bool; abi::Symbol)::Vector{String}
    out = String[]
    if abi === :win64
        return_type in _C_INT128_TYPES &&
            push!(out, "returns a 128-bit integer, which C returns in XMM0 on Win64 where ccall " *
                       "expects a hidden pointer")
        return out
    end
    abi === :sysv || return out
    used = return_in_memory ? 1 : 0
    for (i, t) in enumerate(param_types)
        if t in _C_INT128_TYPES
            used + 2 > 6 &&
                push!(out, "takes a 128-bit integer by value as parameter $i, after arguments that " *
                           "can fill $(min(used, 6)) of the 6 integer registers: C passes it whole, on " *
                           "the stack at a 16-byte boundary, and ccall passes its two halves separately")
            used += 2
        else
            used += gpr_words[i]
        end
    end
    return out
end

"The body of the trap `_c_int128_misplaced` calls for."
_c_int128_trap_body(julia_name::AbstractString, crossings::Vector{String}) =
    "    Base.error(" * repr(
        "ABI Safety Trap: cannot call '$julia_name' through ccall. It $(join(crossings, "; ")). " *
        "The callee would read the value from, or return it to, a place ccall does not use. " *
        "Use a variant of this C function that takes or returns the value through a pointer.") * ")\n"

"""
    handle_unknown_c_type(registry::TypeRegistry, c_type::String, context::String)::String

Handle unknown/unmapped C++ types based on registry strictness settings.

# Behavior
- `STRICT` mode: Error with helpful suggestions
- `WARN` mode: Warn and fallback to Any
- `PERMISSIVE` mode: Silent Any fallback

# Smart fallbacks (when enabled)
- Struct-like types → Use C++ name as opaque struct
- Enum-like types → Map to Cint
- Function pointers → Map to Ptr{Cvoid}
"""
function handle_unknown_c_type(registry::TypeRegistry, c_type::String, context::String)::String
    # Check if it looks like a struct
    if is_c_struct_like(c_type)
        if registry.allow_unknown_structs
            if registry.strictness == WARN
                @warn "Treating unknown type '$c_type' as opaque struct in $context"
            end
            # Remove pointer/reference markers for struct name
            base_type = strip(replace(replace(c_type, "*" => ""), "&" => ""))
            
            # Sanitize for Julia (e.g. Box<int> -> Box_int)
            sanitized = _sanitize_c_type_name(base_type)
            return sanitized
        end
    end

    # Check if it looks like an enum
    if is_c_enum_like(c_type)
        if registry.allow_unknown_enums
            if registry.strictness == WARN
                @warn "Treating enum '$c_type' as Cint in $context"
            end
            return "Cint"
        end
    end

    # Check if it's a function pointer
    if is_c_function_pointer_like(c_type)
        if registry.allow_function_pointers
            if registry.strictness == WARN
                @warn "Treating function pointer '$c_type' as Ptr{Cvoid} in $context"
            end
            return "Ptr{Cvoid}"
        end
    end

    # Apply strictness policy
    if registry.strictness == STRICT
        error("""
        ═══════════════════════════════════════════════════════════════
        FFI Type Mapping Error: Unknown C/C++ Type
        ═══════════════════════════════════════════════════════════════

        Unknown C/C++ type: '$c_type'
        Context: $context

        This type could not be mapped to a Julia type.

        ───────────────────────────────────────────────────────────────
        Suggestions:
        ───────────────────────────────────────────────────────────────

        1. Map it in replibuild.toml:
           [types.custom]
           "$c_type" = "YourJuliaType"

        2. If this is a struct/class:
           - Its definition must reach DWARF: a struct defined only in a header
             needs -fstandalone-debug in [compile] flags
           - Or accept an opaque pointer: [types] allow_unknown_structs = true

        3. If this is an enum:
           - [types] allow_unknown_enums = true maps it to Int32

        4. If this is a function pointer:
           - [types] allow_function_pointers = true maps it to Ptr{Cvoid}

        5. If this is a template type:
           - Name the instantiation in [types] templates, and its header in
             [types] template_headers

        ───────────────────────────────────────────────────────────────
        Temporary Workaround:
        ───────────────────────────────────────────────────────────────

        To allow unmapped types temporarily (not recommended):

           registry = create_c_type_registry(config, strictness=WARN)

        Or for complete permissiveness (legacy mode):

           registry = create_c_type_registry(config, strictness=PERMISSIVE)

        ═══════════════════════════════════════════════════════════════
        """)
    elseif registry.strictness == WARN
        @warn """Unknown C/C++ type '$c_type' in $context, falling back to safe placeholder.
        This function will throw an error if called to prevent GC memory corruption.
        Consider adding a custom type mapping."""
        return "_UnsafeUnknown"
    else  # PERMISSIVE
        return "Any"
    end
end

"""
    infer_c_type(registry::TypeRegistry, c_type::String; context::String="")::String

Infer Julia type from C/C++ type string using comprehensive rules.

# Type Inference Algorithm
1. Strip whitespace and qualifiers (const, volatile)
2. Check exact match in base_types
3. Check exact match in STL types
4. Parse pointer types (T* → Ptr{T})
5. Parse reference types (T& → Ref{T})
6. Parse const pointer types (const T* → Ptr{T})
7. Parse array types (T[N] → NTuple{N,T})
8. Parse template types (std::vector<T> → Vector{T})
9. Smart fallback via handle_unknown_c_type() based on strictness

# Arguments
- `registry`: TypeRegistry with type mappings and validation settings
- `c_type`: C/C++ type string to map
- `context`: Optional context string for error messages (e.g., "parameter 1 of function foo")

# Examples
```julia
infer_c_type(reg, "int") # => "Cint"
infer_c_type(reg, "const char*") # => "Cstring"
infer_c_type(reg, "double*") # => "Ptr{Cdouble}"
infer_c_type(reg, "std::string") # => "String"
infer_c_type(reg, "std::vector<int>") # => "Vector{Cint}"
infer_c_type(reg, "Matrix3x3", context="parameter 1") # => "Matrix3x3" or error in STRICT mode
```
"""
function infer_c_type(registry::TypeRegistry, c_type::String; context::String="")::String
    if isempty(c_type)
        return handle_unknown_c_type(registry, "", context == "" ? "empty type string" : context)
    end

    # Clean up the type string
    clean_type = strip(c_type)

    # Strip const/volatile qualifiers (but remember them)
    is_const = contains(clean_type, "const")
    clean_type = replace(clean_type, r"\bconst\b" => "")
    clean_type = replace(clean_type, r"\bvolatile\b" => "")

    # Strip C11 _Atomic qualifier — atomic types have the same ABI layout
    clean_type = replace(clean_type, r"\b_Atomic\b" => "")

    # Strip C/C++ keywords that might come from DWARF
    clean_type = replace(clean_type, r"\bstruct\b" => "")
    clean_type = replace(clean_type, r"\bunion\b" => "")
    clean_type = replace(clean_type, r"\bclass\b" => "")

    clean_type = strip(clean_type)

    if clean_type in _INTERNAL_TYPE_BLOCKLIST
        return "Cvoid"
    end

    # Handle special case: const char* and char* are Cstring
    if clean_type == "char*" || clean_type == "char *"
        return "Cstring"
    end

    # Check exact match in base types
    if haskey(registry.base_types, clean_type)
        return registry.base_types[clean_type]
    end

    # Check custom types
    if haskey(registry.custom_types, clean_type)
        return registry.custom_types[clean_type]
    end

    # Parse pointer types: T* or T *
    if endswith(clean_type, "*")
        base_type = strip(replace(clean_type, r"\*$" => ""))

        # Special cases
        if base_type == "void"
            return "Ptr{Cvoid}"
        elseif base_type == "char"
            return "Cstring"
        end

        # Recursive inference for base type
        ptr_context = context == "" ? "pointer base type" : "$context (pointer to $base_type)"
        julia_base = infer_c_type(registry, String(base_type); context=ptr_context)
        return "Ptr{$julia_base}"
    end

    # Parse reference types: T& or T &
    if endswith(clean_type, "&")
        base_type = strip(replace(clean_type, r"&$" => ""))
        ref_context = context == "" ? "reference base type" : "$context (reference to $base_type)"
        julia_base = infer_c_type(registry, String(base_type); context=ref_context)
        return "Ref{$julia_base}"
    end

    # Parse array types: T[N]
    array_match = match(r"^(.+)\[(\d+)\]$", clean_type)
    if !isnothing(array_match)
        elem_type = strip(array_match.captures[1])
        size = parse(Int, array_match.captures[2])
        arr_context = context == "" ? "array element type" : "$context (array of $elem_type)"
        julia_elem = infer_c_type(registry, String(elem_type); context=arr_context)
        return "NTuple{$size,$julia_elem}"
    end

    # GCC/Clang vector types: __attribute__((vector_size(N))) or __vector(N) T
    # DWARF emits these as e.g. "__vector(4) float" or "float __attribute__((__vector_size__(16)))"
    vec_match = match(r"^__vector\((\d+)\)\s+(.+)$", clean_type)
    if !isnothing(vec_match)
        vec_len = parse(Int, vec_match.captures[1])
        elem = strip(vec_match.captures[2])
        julia_elem = infer_c_type(registry, String(elem); context="$context (vector element)")
        return "NTuple{$vec_len,$julia_elem}"
    end
    # Alternative DWARF format: "T __attribute__((__vector_size__(N)))"
    vec_attr_match = match(r"^(.+?)\s*__attribute__\(\(__vector_size__\((\d+)\)\)\)$", clean_type)
    if !isnothing(vec_attr_match)
        elem = strip(vec_attr_match.captures[1])
        total_bytes = parse(Int, vec_attr_match.captures[2])
        julia_elem = infer_c_type(registry, String(elem); context="$context (vector element)")
        # vector_size is in bytes; compute element count from element size
        elem_size = get(Dict("float" => 4, "double" => 8, "int" => 4, "short" => 2, "char" => 1,
                             "long" => 8, "long long" => 8), elem, 4)
        vec_len = max(1, total_bytes ÷ elem_size)
        return "NTuple{$vec_len,$julia_elem}"
    end

    # Unknown type - no matching rules
    ctx = context == "" ? "type inference" : context
    return handle_unknown_c_type(registry, String(clean_type), ctx)
end

