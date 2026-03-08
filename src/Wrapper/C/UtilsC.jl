# =============================================================================
# C UTILITY FUNCTIONS
# =============================================================================

# Helper: sanitize a C type name to a valid Julia struct/type identifier
function _sanitize_c_type_name(name::AbstractString)::String
    s = replace(string(name), " "  => "")
    s = replace(s, "-"  => "minus_")
    s = replace(s, "+"  => "plus_")
    s = replace(s, "*"  => "star_")
    # Collapse consecutive underscores and trim trailing ones
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
