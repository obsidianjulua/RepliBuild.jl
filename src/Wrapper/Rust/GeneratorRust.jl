# =============================================================================
# RUST WRAPPER GENERATOR
# =============================================================================

import Dates
using JSON

# ─── Rust stdlib DWARF leakage filter ────────────────────────────────────────

const _RUST_STDLIB_BLOCKLIST = Set([
    # core::fmt
    "Formatter", "FormattingOptions", "Placeholder", "Arguments", "Argument",
    "ArgumentType", "Count",
    # core::result / core::option / core::ops
    "Ok", "Err", "Some", "None", "Continue", "Break",
    "Result", "Option", "Error",
    # core::alloc / alloc::alloc
    "Layout", "AllocError", "Alignment", "NonZeroUsizeInner", "UsizeNoHighBit",
    "TryReserveError", "TryReserveErrorKind",
    # alloc::ffi / core::ffi
    "NulError", "CString", "CStr",
    # alloc::string / alloc::vec
    "String", "RawVecInner", "RawVec",
    # core::ptr
    "NonNull", "Unique",
    # core::panic
    "Location",
    # core::str
    "Utf8Error",
    # std::io::error internals
    "Custom", "Os", "Repr", "Simple", "SimpleMessage", "ErrorData", "ErrorKind",
])

"""
    _is_rust_stdlib_type(name::String)::Bool

Returns true if `name` is a Rust stdlib/core type that leaked through DWARF
and should not be emitted as a user-facing Julia struct.
"""
function _is_rust_stdlib_type(name::String)::Bool
    # 1. Namespaced types (core::*, alloc::*, std::*)
    if occursin("core::", name) || occursin("alloc::", name) || occursin("std::", name)
        return true
    end
    # 2. Module-qualified paths (anything with ::)
    if occursin("::", name)
        return true
    end
    # 3. Generic instantiations (NonNull<T>, Vec<T, A>, Result<T, E>, etc.)
    if occursin('<', name)
        return true
    end
    # 4. Closure environments
    if occursin("{closure", name) || occursin("closure_env", name)
        return true
    end
    # 5. Vtable types
    if occursin("vtable_type", name)
        return true
    end
    # 6. Tuple types
    if startswith(name, "(")
        return true
    end
    # 7. Reference/slice types leaked from DWARF
    if startswith(name, "&") || startswith(name, "*const ") || startswith(name, "*mut ")
        return true
    end
    # 8. Direct blocklist (stripped of module path for matching)
    if name in _RUST_STDLIB_BLOCKLIST
        return true
    end
    # 9. Blocklist with suffix stripping (e.g., AlignmentEnum matches Alignment)
    for blocked in _RUST_STDLIB_BLOCKLIST
        if startswith(name, blocked)
            return true
        end
    end
    return false
end

# ─── Generator ───────────────────────────────────────────────────────────────

"""
    generate_introspective_module_rust(config::RepliBuildConfig, lib_path::String, metadata, module_name::String, registry::TypeRegistry, generate_docs::Bool, thunks_lib_path::String="")

Generate an introspective Julia wrapper module for a Rust library.
Uses DWARF debug metadata for struct layouts, enum values, and function signatures.
"""
function generate_introspective_module_rust(config::RepliBuildConfig, lib_path::String,
                                      metadata, module_name::String,
                                      registry::TypeRegistry, generate_docs::Bool,
                                      thunks_lib_path::String="")

    exports = String[]

    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Rust Introspective)
    # Library: $(basename(lib_path))
    # Metadata: compilation_metadata.json

    module $module_name

    const Cintptr_t = Int
    const Cuintptr_t = UInt
    const Cssize_t = Int
    const Csize_t = UInt

    using Libdl
    import RepliBuild
    import Base: unsafe_convert

    const LIBRARY_PATH = "$(abspath(lib_path))"
    const THUNKS_LIBRARY_PATH = "$(thunks_lib_path)"

    # Verify library exists
    if !isfile(LIBRARY_PATH)
        error("Library not found: \$LIBRARY_PATH")
    end

    """

    content = header

    # ─── Extract and classify DWARF types ────────────────────────────────
    all_structs = get(metadata, "struct_definitions", Dict{String,Any}())
    dwarf_structs = Dict{String,Any}()
    dwarf_enums = Dict{String,Any}()
    for (k, v) in all_structs
        if startswith(k, "__enum__")
            enum_name = replace(k, "__enum__" => "")
            if !_is_rust_stdlib_type(enum_name)
                dwarf_enums[enum_name] = v
            end
        else
            if !_is_rust_stdlib_type(k)
                dwarf_structs[k] = v
            end
        end
    end
    functions = get(metadata, "functions", [])

    # Collect the set of known user struct names (sanitized)
    struct_julia_names = Dict{String,String}()  # julia_name => original_name
    for (name, _) in dwarf_structs
        jname = make_rust_identifier(name)
        if !haskey(struct_julia_names, jname)
            struct_julia_names[jname] = name
        end
    end

    # ─── Build dependency graph for topological sort ─────────────────────
    struct_deps = Dict{String,Set{String}}()
    for (name, info) in dwarf_structs
        jname = make_rust_identifier(name)
        deps = Set{String}()
        members = get(info, "members", [])
        for member in members
            m_julia = _resolve_member_type(registry, member)
            # Extract base type from Ptr{X} wrappers
            base_ref = m_julia
            while startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
                base_ref = base_ref[5:end-1]
            end
            # If it references another user struct, add dependency
            # Only add hard dep for by-value (non-Ptr) references
            if base_ref in keys(struct_julia_names) && base_ref != jname
                if m_julia == base_ref  # by-value, not Ptr{...}
                    push!(deps, base_ref)
                end
                # Ptr{X} is a soft dependency — no ordering constraint
            end
        end
        struct_deps[jname] = deps
    end

    # Kahn's algorithm for topological sort
    sorted_struct_names = String[]
    remaining = Dict(k => copy(v) for (k, v) in struct_deps)

    while !isempty(remaining)
        ready = [name for (name, deps) in remaining if isempty(deps)]
        if isempty(ready)
            # Circular dependency — take alphabetically first to break cycle
            ready = [sort(collect(keys(remaining)))[1]]
        end
        for name in sort(ready)
            push!(sorted_struct_names, name)
            delete!(remaining, name)
            for deps in values(remaining)
                delete!(deps, name)
            end
        end
    end

    # ─── Detect opaque types referenced in function signatures ───────────
    # Types referenced via Ptr{X} in functions but not defined as structs
    opaque_types = Set{String}()
    defined_struct_names = Set(sorted_struct_names)

    for func in functions
        _collect_ptr_refs!(opaque_types, defined_struct_names, func, registry)
    end
    # Also scan struct members for forward-referenced pointer types
    for (name, info) in dwarf_structs
        for member in get(info, "members", [])
            m_julia = _resolve_member_type(registry, member)
            base_ref = m_julia
            while startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
                base_ref = base_ref[5:end-1]
            end
            if !(base_ref in defined_struct_names) && base_ref != make_rust_identifier(name) &&
               !_is_julia_builtin(base_ref) && startswith(m_julia, "Ptr{")
                push!(opaque_types, base_ref)
            end
        end
    end

    # ─── Emit forward declarations for opaque types ──────────────────────
    seen_struct_names = Set{String}()
    if !isempty(opaque_types)
        content *= "# =============================================================================\n"
        content *= "# OPAQUE TYPES (forward declarations)\n"
        content *= "# =============================================================================\n\n"
        for oname in sort(collect(opaque_types))
            if !(oname in seen_struct_names)
                content *= "mutable struct $oname end\n"
                push!(seen_struct_names, oname)
                push!(exports, oname)
            end
        end
        content *= "\n"
    end

    # ─── Emit structs in topological order ───────────────────────────────
    if !isempty(sorted_struct_names)
        content *= "# =============================================================================\n"
        content *= "# STRUCTS\n"
        content *= "# =============================================================================\n\n"

        for jname in sorted_struct_names
            if jname in seen_struct_names
                continue  # dedup
            end
            push!(seen_struct_names, jname)

            orig_name = get(struct_julia_names, jname, jname)
            info = get(dwarf_structs, orig_name, nothing)
            if isnothing(info)
                continue
            end

            members = get(info, "members", [])
            if isempty(members)
                # Opaque type — no layout known
                content *= "mutable struct $jname end\n"
                push!(exports, jname)
                continue
            end

            content *= "struct $jname\n"
            for member in members
                m_name = make_rust_identifier(get(member, "name", "unknown"))
                m_julia_type = _resolve_member_type(registry, member)
                content *= "    $m_name::$m_julia_type\n"
            end
            content *= "end\n\n"
            push!(exports, jname)
        end
    end

    # ─── Emit enums ──────────────────────────────────────────────────────
    if !isempty(dwarf_enums)
        content *= "# =============================================================================\n"
        content *= "# ENUMS\n"
        content *= "# =============================================================================\n\n"

        for (name, info) in sort(collect(dwarf_enums), by=first)
            enumerators = get(info, "enumerators", get(info, "members", []))
            if isempty(enumerators)
                continue
            end

            julia_name = make_rust_identifier(name)

            # Use underlying type from DWARF metadata
            underlying = get(info, "underlying_type", "i32")
            julia_underlying = infer_rust_type(registry, underlying, context="Enum $name underlying")
            # Validate it's a valid @enum base type
            valid_enum_bases = Set(["Int8", "UInt8", "Int16", "UInt16", "Int32",
                                     "UInt32", "Int64", "UInt64"])
            if !(julia_underlying in valid_enum_bases)
                julia_underlying = "Int32"
            end

            is_unsigned = startswith(julia_underlying, "UInt")
            content *= "@enum $julia_name::$julia_underlying begin\n"
            for e in enumerators
                e_name = make_rust_identifier(get(e, "name", "UNKNOWN"))
                e_val_raw = get(e, "value", "0")
                # Handle negative DWARF values for unsigned enum types
                e_val = try
                    v = parse(Int128, string(e_val_raw))
                    if is_unsigned && v < 0
                        string(reinterpret(UInt64, Int64(v)))
                    else
                        string(v)
                    end
                catch
                    string(e_val_raw)
                end
                content *= "    $(e_name) = $e_val\n"
            end
            content *= "end\n\n"
            push!(exports, julia_name)
        end
    end

    # ─── Emit function wrappers ──────────────────────────────────────────
    if !isempty(functions)
        content *= "# =============================================================================\n"
        content *= "# FUNCTIONS\n"
        content *= "# =============================================================================\n\n"

        seen_functions = Set{String}()

        for func in functions
            raw_name = get(func, "name", "")
            julia_name = make_rust_identifier(raw_name)

            # Skip empty, internal, and duplicate symbols
            if isempty(raw_name) || startswith(raw_name, "_")
                continue
            end
            if julia_name in seen_functions
                continue
            end
            push!(seen_functions, julia_name)

            # Return type — prefer DWARF julia_type, fall back to type inference
            ret_info = get(func, "return_type", Dict())
            c_ret_type = get(ret_info, "c_type", "void")
            julia_ret_type = _resolve_type(registry, ret_info, c_ret_type,
                                            context="Return type of $raw_name")

            # Parameters
            params = get(func, "parameters", [])
            param_names = String[]
            param_types = String[]

            for (i, p) in enumerate(params)
                p_name = get(p, "name", "arg$i")
                p_c_type = get(p, "c_type", "void")
                p_julia_name = make_rust_identifier(p_name)
                p_julia_type = _resolve_type(registry, p, p_c_type,
                                              context="Parameter $p_name of $raw_name")

                push!(param_names, p_julia_name)
                push!(param_types, p_julia_type)
            end

            # Documentation
            if generate_docs
                content *= "\"\"\"\n"
                content *= "    $julia_name($(join(param_names, ", ")))\n\n"
                content *= "Wrapper for Rust function `$raw_name`.\n"
                content *= "\"\"\"\n"
            end

            # Function body — Tier 3 ccall (Tier 1 llvmcall added in Phase 2)
            sig = join(["$(n)::$(t)" for (n, t) in zip(param_names, param_types)], ", ")
            ccall_types = isempty(param_types) ? "()" : "($(join(param_types, ", ")),)"
            ccall_args = isempty(param_names) ? "" : ", $(join(param_names, ", "))"

            content *= "function $julia_name($sig)::$julia_ret_type\n"
            content *= "    return ccall((:$raw_name, LIBRARY_PATH), $julia_ret_type, $ccall_types$ccall_args)\n"
            content *= "end\n\n"

            push!(exports, julia_name)
        end
    end

    # ─── Exports ─────────────────────────────────────────────────────────
    content *= "# =============================================================================\n"
    content *= "# EXPORTS\n"
    content *= "# =============================================================================\n\n"
    content *= "export " * join(unique(exports), ", ") * "\n\n"

    content *= "end # module $module_name\n"

    return content
end

# ─── Helper: resolve a member's Julia type from metadata ─────────────────────

"""
Resolve a struct member's Julia type from DWARF metadata.
Always uses `infer_rust_type(c_type)` as the primary path since DWARF `c_type`
is ground truth for Rust types (the metadata `julia_type` may have been mapped
through the C/C++ type system incorrectly).
"""
function _resolve_member_type(registry::TypeRegistry, member)::String
    c_type = get(member, "c_type", "void")
    return infer_rust_type(registry, c_type, context="member")
end

"""
Resolve a type from metadata dict entry using Rust type inference.
"""
function _resolve_type(registry::TypeRegistry, info, c_type::String; context::String="")::String
    return infer_rust_type(registry, c_type, context=context)
end

# ─── Helper: collect Ptr{X} references from function signatures ─────────────

function _collect_ptr_refs!(opaque::Set{String}, defined::Set{String}, func, registry::TypeRegistry)
    # Check return type
    ret_info = get(func, "return_type", Dict())
    c_ret = get(ret_info, "c_type", "void")
    _check_ptr_ref!(opaque, defined, infer_rust_type(registry, c_ret))

    # Check parameters
    for p in get(func, "parameters", [])
        p_type = get(p, "c_type", "void")
        _check_ptr_ref!(opaque, defined, infer_rust_type(registry, p_type))
    end
end

function _check_ptr_ref!(opaque::Set{String}, defined::Set{String}, julia_type::String)
    base_ref = julia_type
    while startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
        base_ref = base_ref[5:end-1]
    end
    if !(base_ref in defined) && !_is_julia_builtin(base_ref) && base_ref != julia_type
        push!(opaque, base_ref)
    end
end

function _is_julia_builtin(t::String)::Bool
    t in Set(["Cvoid", "Cint", "Cuint", "Cchar", "Cuchar", "Cshort", "Cushort",
              "Clong", "Culong", "Clonglong", "Culonglong", "Cfloat", "Cdouble",
              "Csize_t", "Cssize_t", "Cptrdiff_t", "Cintptr_t", "Cuintptr_t",
              "Bool", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32",
              "Int64", "UInt64", "Int128", "UInt128", "Float32", "Float64",
              "Cstring", "Nothing", "Any", "UInt8"])
end
