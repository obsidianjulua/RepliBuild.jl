# =============================================================================
# Dispatch logic — Per-function tier routing decisions
# =============================================================================
# Separated from Utils.jl because is_ccall_safe() depends on
# is_stl_container_type() (defined in Cpp/TypesCpp.jl), which is included
# later in the load order.  is_c_lto_safe() has no such dependency but lives
# here to keep all dispatch decisions in one file.

"""
    _is_enum_type(type_name, dwarf_structs) -> Bool

Check if a type name corresponds to an enum in DWARF metadata.
Enums are stored with `__enum__` prefix (e.g. `__enum__MathOp`).
Enums are integers at the ABI level — always safe for ccall.
"""
function _is_enum_type(type_name::AbstractString, dwarf_structs)::Bool
    return haskey(dwarf_structs, "__enum__$(type_name)")
end

# =============================================================================
# is_c_lto_safe() — C-specific dispatch gate
# =============================================================================

"""
    is_c_lto_safe(func_info, dwarf_structs)::Bool

Determine if a C function can be dispatched via direct ccall/llvmcall.
Returns false (→ route to Clang-compiled sret thunk) only when the
**return type** is:

- A packed struct (DWARF byte_size ≠ Julia aligned size)
- A union by value

C has no templates, vtables, STL, or inheritance — the only ABI hazards
are packed struct returns and union returns where ccall's return convention
doesn't match the actual layout.

Parameters by value are fine: the caller pushes bytes according to the ABI
regardless of packing, so ccall handles them correctly.
"""
function is_c_lto_safe(func_info, dwarf_structs)::Bool
    ret_type = String(get(func_info["return_type"], "c_type", ""))

    # Void, pointer, primitive returns — always safe
    if ret_type == "void" || contains(ret_type, "*") || _is_primitive_type(ret_type)
        return true
    end

    # Enum returns are integers at the ABI level — safe
    cleaned_ret = strip(replace(ret_type, r"\bconst\b" => ""))
    if _is_enum_type(cleaned_ret, dwarf_structs)
        return true
    end

    # Check struct return safety
    if haskey(dwarf_structs, cleaned_ret)
        s_info = dwarf_structs[cleaned_ret]

        # Union return — ccall can't handle by-value union returns
        if get(s_info, "kind", "struct") == "union"
            return false
        end

        # Packed struct return: DWARF byte_size ≠ Julia aligned size
        dwarf_size = _parse_dwarf_size(s_info)
        members = get(s_info, "members", [])

        # Resolve nested struct member sizes (same fix as _is_struct_unsafe)
        resolved_members = map(members) do m
            if get(m, "size", 0) == 0
                m_type = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
                if haskey(dwarf_structs, m_type)
                    nested_size = _parse_dwarf_size(dwarf_structs[m_type])
                    if nested_size > 0
                        m2 = copy(m)
                        m2["size"] = nested_size
                        return m2
                    end
                end
            end
            return m
        end

        julia_size = get_julia_aligned_size(resolved_members)
        if dwarf_size > 0 && julia_size > 0 && dwarf_size != julia_size
            return false
        end
    end

    return true
end

"""
    is_ccall_safe(func_info, dwarf_structs)::Bool

Determine if a function is safe for standard `ccall`.
Returns false (→ route to MLIR Tier 2) if any of:

**Return type:**
- STL container by value
- Template type (contains `<`)
- Packed struct (DWARF size ≠ Julia size)
- Overaligned (alignment > 8)
- Polymorphic (has vtable)
- Inherits from base class
- Non-POD class with members
- Contains unsafe nested struct member
- Unknown struct not in DWARF (can't verify safety)

**Parameters (by value, not pointer/ref):**
- STL container
- Template type (contains `<`)
- Union
- Packed struct
- Overaligned
- Polymorphic / inherits
- Contains unsafe nested member
- Unknown struct not in DWARF
"""
function is_ccall_safe(func_info, dwarf_structs)
    # ── 0. STL containers (never ccall-safe by value) ───────────────────
    ret_type_str = String(get(func_info["return_type"], "c_type", ""))
    if is_stl_container_type(ret_type_str)
        return false
    end
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")
        if contains(c_type, "&") || contains(c_type, "*")
            continue
        end
        if is_stl_container_type(c_type)
            return false
        end
    end

    # ── 1. Return type analysis ─────────────────────────────────────────
    ret_type = ret_type_str

    # Quick exit: pointer, void, or primitive returns are safe
    is_template_ret = occursin('<', ret_type)
    is_primitive_ret = !is_template_ret && _is_primitive_type(ret_type)
    is_pointer_ret = contains(ret_type, "*")
    is_void_ret = ret_type == "void" || strip(replace(ret_type, r"\bconst\b" => "")) == "void"

    if !is_pointer_ret && !is_void_ret && !is_primitive_ret
        # Template return types — always route to MLIR.
        # DWARF may not have the exact specialization.
        if is_template_ret
            return false
        end

        # Enums are integers at the ABI level — always safe
        cleaned_ret = strip(replace(ret_type, r"\bconst\b" => ""))
        if _is_enum_type(cleaned_ret, dwarf_structs)
            # safe, fall through
        elseif haskey(dwarf_structs, ret_type)
            s_info = dwarf_structs[ret_type]

            # Full struct safety check (packed, overaligned, polymorphic, etc.)
            if _is_struct_unsafe(s_info, dwarf_structs)
                return false
            end
        else
            # Return type is a struct name but NOT in DWARF — can't verify safety.
            # If it looks like an identifier (not a known primitive), route to MLIR.
            if occursin(r"^[A-Za-z_][A-Za-z0-9_:]*$", cleaned_ret) && !_is_primitive_type(cleaned_ret)
                return false
            end
        end
    end

    # ── 2. Parameter analysis ───────────────────────────────────────────
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")

        # Pointers and references are always safe at ABI level
        if contains(c_type, "*") || contains(c_type, "&")
            continue
        end

        # Template parameter by value — always route to MLIR
        if occursin('<', c_type)
            return false
        end

        # Clean const prefix for base type lookup
        base_type = String(strip(replace(c_type, r"\bconst\b" => "")))

        # Enums are integers at the ABI level — always safe
        if _is_enum_type(base_type, dwarf_structs)
            continue
        end

        if haskey(dwarf_structs, base_type)
            s_info = dwarf_structs[base_type]

            # Full struct safety check
            if _is_struct_unsafe(s_info, dwarf_structs)
                return false
            end
        else
            # Unknown struct parameter not in DWARF — can't verify safety
            cleaned = strip(base_type)
            if occursin(r"^[A-Za-z_][A-Za-z0-9_:]*$", cleaned) && !_is_primitive_type(cleaned)
                return false
            end
        end
    end

    return true
end
