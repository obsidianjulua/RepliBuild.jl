# =============================================================================
# C++ UTILITY FUNCTIONS
# =============================================================================

# Helper: sanitize a C++ type name to a valid Julia struct/type identifier
function _sanitize_cpp_type_name(name::AbstractString)::String
    s = replace(string(name), "::" => "_")
    s = replace(s, "<"  => "_")
    s = replace(s, ">"  => "")
    s = replace(s, ","  => "_")
    s = replace(s, " "  => "")
    s = replace(s, "-"  => "minus_")
    s = replace(s, "+"  => "plus_")
    s = replace(s, "*"  => "star_")
    # Collapse consecutive underscores and trim trailing ones so that every
    # call-site (struct definitions, field types, function parameters, …)
    # produces identical identifiers for the same C++ type.
    s = replace(s, r"_+" => "_")
    s = String(rstrip(s, '_'))
    if !isempty(s) && isdigit(s[1])
        s = "_" * s
    end
    if s in ("for", "if", "else", "while", "function", "struct", "end", "module", "using", "import", "export", "return", "continue", "break", "try", "catch", "finally", "macro", "quote", "let", "local", "global", "const", "do", "baremodule", "true", "false", "abstract", "type", "mutable", "primitive")
        s = "c_" * s
    end
    return s
end

"""Normalize an inferred Julia type string to a ccall-compatible type alias."""
function _normalize_stl_elem_type(jtype::String)::String
    if jtype in ("Cint", "Int32")
        return "Cint"
    elseif jtype in ("Cdouble", "Float64")
        return "Cdouble"
    elseif jtype in ("Cfloat", "Float32")
        return "Cfloat"
    elseif jtype == "Int64"
        return "Int64"
    elseif jtype == "UInt64"
        return "UInt64"
    end
    return jtype
end

"""
    _fuzzy_dwarf_lookup(c_type, dwarf_structs) -> Union{String, Nothing}

Fuzzy-match a C type name against DWARF struct definition keys.
DWARF keys for template types often have trailing " >" nesting artifacts.
"""
function _fuzzy_dwarf_lookup(c_type::AbstractString, dwarf_structs)
    haskey(dwarf_structs, c_type) && return String(c_type)
    key = c_type * " >"
    haskey(dwarf_structs, key) && return key
    c_norm = rstrip(c_type, [' ', '>'])
    for k in keys(dwarf_structs)
        rstrip(String(k), [' ', '>']) == c_norm && return String(k)
    end
    return nothing
end
