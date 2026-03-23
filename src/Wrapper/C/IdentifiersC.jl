# =============================================================================
# C IDENTIFIER GENERATION
# =============================================================================

"""
    make_c_identifier(name::String)::String

Convert C symbol to valid Julia identifier.
"""
function make_c_identifier(name::String)::String
    if isempty(name)
        return ""
    end

    clean = replace(name, r"[^a-zA-Z0-9_!]" => "_")
    clean = replace(clean, r"_{2,}" => "_")

    if !isempty(clean) && isdigit(clean[1])
        clean = "_" * clean
    end

    julia_keywords = [
        "begin", "end", "if", "else", "elseif", "while", "for", "function",
        "return", "break", "continue", "module", "using", "import", "export",
        "struct", "mutable", "abstract", "type", "const", "global", "local",
        "let", "do", "try", "catch", "finally", "macro", "quote", "true",
        "false", "nothing", "missing", "NaN", "Inf"
    ]

    if clean in julia_keywords
        clean = clean * "_"
    end

    return clean
end
