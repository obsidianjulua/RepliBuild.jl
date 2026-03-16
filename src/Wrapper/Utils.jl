
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Complete set of Julia reserved keywords and soft keywords for enum member escaping
const _JULIA_KEYWORDS = Set([
    "baremodule", "begin", "break", "catch", "const", "continue", "do",
    "else", "elseif", "end", "export", "false", "finally", "for",
    "function", "global", "if", "import", "in", "let", "local",
    "macro", "module", "mutable", "nothing", "quote", "return",
    "struct", "true", "try", "type", "using", "while", "abstract",
    "primitive", "where", "isa",
    # Not technically reserved but conflict as identifiers in enum context
    "and", "or", "not",
])

# Internal/compiler types that leak through DWARF but shouldn't be exported
const _INTERNAL_TYPE_BLOCKLIST = Set([
    "__va_list_tag", "__mbstate_t", "__loadu_pd", "__storeu_pd",
    "__loadu_ps", "__storeu_ps", "__loadu_si128", "__storeu_si128",
    "_va_list_tag", "_mbstate_t", "_loadu_pd", "_storeu_pd",
    "_loadu_ps", "_storeu_ps",
    "ldiv_t", "lldiv_t", "div_t", "max_align_t", "imaxdiv_t",
    "_IO_FILE", "_IO_marker", "_IO_codecvt", "_IO_wide_data",
])

"""Escape a name if it's a Julia keyword, using var\"...\" syntax."""
function _escape_keyword(name::String)::String
    if name in _JULIA_KEYWORDS
        return "var\"$name\""
    end
    return name
end

# Built-in Julia types that never need forward declaration
const _JULIA_BUILTIN_TYPES = Set([
    "Cvoid", "Cint", "Cuint", "Clong", "Culong", "Cshort", "Cushort",
    "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
    "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
    "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
    "Cstring", "Float32", "Float64", "Any", "Nothing", "Cintptr_t", "Cuintptr_t",
])

"""
    _resolve_forward_ptr(julia_type, defined_names) -> String

Replace `Ptr{X}` (including nested `Ptr{Ptr{X}}`) with `Ptr{Cvoid}` when `X`
is a custom struct type that has not been defined yet. This avoids
`UndefVarError` from forward references while preserving ABI layout
(all pointers are the same size).
"""
function _resolve_forward_ptr(julia_type::AbstractString, defined_names::Set{String})::String
    m = match(r"^Ptr\{(.+)\}$", julia_type)
    isnothing(m) && return julia_type
    inner = m.captures[1]
    # Recurse for nested Ptr
    resolved_inner = _resolve_forward_ptr(inner, defined_names)
    # Check if the innermost type is defined or builtin
    base = inner
    while (bm = match(r"^Ptr\{(.+)\}$", base)) !== nothing
        base = bm.captures[1]
    end
    if base in _JULIA_BUILTIN_TYPES || base in defined_names
        return "Ptr{$resolved_inner}"
    end
    return "Ptr{Cvoid}"
end

# =============================================================================
# DISPATCH LOGIC HELPERS
# =============================================================================

"""
    get_julia_aligned_size(members::Vector)

Calculate the size of a struct in Julia including standard padding alignment.
Used to detect if a C++ struct is 'packed' (Julia size > DWARF size).
"""
function get_julia_aligned_size(members::Vector)
    current_offset = 0
    max_align = 1

    for m in members
        # specific size of this member
        m_size = get(m, "size", 0)

        # simple alignment heuristic (size usually equals alignment for primitives)
        # generic pointer/int alignment cap at 8 bytes on 64-bit
        align = m_size > 8 ? 8 : m_size
        align = align == 0 ? 1 : align # handle empty/void

        # update generic alignment requirement
        max_align = max(max_align, align)

        # Add padding to current offset
        padding = (align - (current_offset % align)) % align
        current_offset += padding + m_size
    end

    # Final structure alignment padding
    padding = (max_align - (current_offset % max_align)) % max_align
    return current_offset + padding
end

"""
    _parse_dwarf_size(s_info) -> Int

Parse byte_size from a DWARF struct info dict, handling both decimal and hex.
"""
function _parse_dwarf_size(s_info)::Int
    raw = get(s_info, "byte_size", "0")
    s = raw isa String ? raw : string(raw)
    try
        startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s)
    catch
        0
    end
end

"""
    _is_struct_unsafe(s_info, dwarf_structs) -> Bool

Check if a struct type is unsafe for ccall (should route to MLIR).
Uses all available DWARF metadata: size, alignment, packing, polymorphism,
inheritance, and member types.
"""
function _is_struct_unsafe(s_info, dwarf_structs)::Bool
    # Packed struct: DWARF size != Julia calculated size (alignment mismatch)
    dwarf_size = _parse_dwarf_size(s_info)
    members = get(s_info, "members", [])

    # Resolve member sizes: when a member's c_type is itself a struct in
    # dwarf_structs and the DWARF parser left its size=0, substitute the
    # nested struct's byte_size so get_julia_aligned_size computes correctly.
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
        return true
    end

    # Overaligned struct: alignment > 8 means SIMD/cache-line alignment
    # that ccall won't respect (Julia uses natural alignment capped at 8)
    alignment = get(s_info, "alignment", 0)
    alignment = alignment isa String ? (try parse(Int, alignment) catch; 0 end) : alignment
    if alignment > 8
        return true
    end

    # Union: layout is overlapping, ccall can't handle by-value unions
    if get(s_info, "kind", "struct") == "union"
        return true
    end

    # Polymorphic: has vtable pointer, non-trivial copy/move semantics
    if get(s_info, "is_polymorphic", false) == true
        return true
    end

    # Inherits from another class: likely non-trivial layout (vptr, padding)
    base_classes = get(s_info, "base_classes", [])
    if !isempty(base_classes)
        return true
    end

    # Class keyword: C++ class defaults to private members, often non-POD.
    # Only flag if it also has members (empty tag classes are safe).
    if get(s_info, "kind", "struct") == "class" && !isempty(members)
        return true
    end

    # Member contains a nested struct that is itself unsafe
    for m in members
        m_type = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
        # Skip pointers and references — those are just addresses
        if endswith(m_type, "*") || endswith(m_type, "&") || contains(m_type, "*")
            continue
        end
        if haskey(dwarf_structs, m_type)
            nested = dwarf_structs[m_type]
            # Recursive check (one level deep to avoid cycles)
            if _is_struct_unsafe(nested, Dict{String,Any}())
                return true
            end
        end
    end

    return false
end

"""
Canonical set of C/C++ primitive type names that are safe for ccall.
Used by `is_ccall_safe()` to distinguish primitives from struct types
without fragile substring matching (e.g. `contains(ret_type, "int")`
would match "Point").
"""
const _CCALL_SAFE_PRIMITIVES = Set([
    "void", "int", "unsigned int", "signed int",
    "char", "unsigned char", "signed char",
    "short", "unsigned short", "short int",
    "long", "unsigned long", "long int", "long long", "unsigned long long",
    "float", "double", "long double",
    "bool", "_Bool",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t", "wchar_t",
])

"""
    _is_primitive_type(c_type::AbstractString) -> Bool

Check if a C type string (after stripping const/volatile/whitespace) is a
known primitive. Uses `_CCALL_SAFE_PRIMITIVES` set for O(1) lookup.
"""
function _is_primitive_type(c_type::AbstractString)::Bool
    cleaned = strip(replace(c_type, r"\b(const|volatile|restrict)\b" => ""))
    cleaned = strip(replace(cleaned, r"\s+" => " "))
    return cleaned in _CCALL_SAFE_PRIMITIVES
end

# is_ccall_safe() is defined in Wrapper/DispatchLogic.jl (included after TypesCpp.jl
# to have access to is_stl_container_type)

