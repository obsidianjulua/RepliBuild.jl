# =============================================================================
# RUST IDENTIFIER GENERATION
# =============================================================================

"""
    make_rust_identifier(name::AbstractString)::String

Convert Rust symbol to valid Julia identifier.
Rust identifiers are usually very clean compared to C++ mangled names,
but we still need to avoid Julia keywords.
"""
function make_rust_identifier(name::AbstractString)::String
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

    if lowercase(clean) in julia_keywords
        clean = clean * "_"
    end

    return clean
end
