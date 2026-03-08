
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

