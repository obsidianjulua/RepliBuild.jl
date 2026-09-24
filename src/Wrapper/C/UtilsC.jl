# =============================================================================
# C UTILITY FUNCTIONS
# =============================================================================

# Helper: sanitize a C type name to a valid Julia struct/type identifier
function _sanitize_c_type_name(name::AbstractString)::String
    s = replace(string(name), " "  => "_")
    s = replace(s, "-"  => "minus_")
    s = replace(s, "+"  => "plus_")
    s = replace(s, "*"  => "star_")
    # Collapse consecutive underscores and trim trailing ones
    s = replace(s, r"_+" => "_")
    # An all-underscore name is a DIFFERENT case from a genuinely empty one and
    # must not share its answer. This function sanitizes enum MEMBER names too
    # (GeneratorC.jl:663), so a C enum spelled `{ _, … }` would otherwise be
    # renamed `_UnknownType` — readable, but silently wrong, and the name no
    # longer matches the header the user is reading from. Restore it as `c__`,
    # the same escape the keyword branch below uses. Plain `_` is not an option:
    # Julia's all-underscore identifiers are write-only, so the member could
    # never be referenced. See the long note in
    # `_sanitize_cpp_type_name` (Wrapper/Cpp/UtilsCpp.jl) — the C++ side had no
    # empty guard at all and emitted invalid Julia; these two must stay in step.
    was_all_underscore = !isempty(s) && all(==('_'), s)
    s = String(rstrip(s, '_'))
    was_all_underscore && (s = "c__")
    if !isempty(s) && isdigit(s[1])
        s = "_" * s
    end
    if isempty(s)
        return "_UnknownType"
    end
    if s in ("for", "if", "else", "while", "function", "struct", "end", "module", "using", "import", "export", "return", "continue", "break", "try", "catch", "finally", "macro", "quote", "let", "local", "global", "const", "do", "baremodule", "true", "false", "abstract", "type", "mutable", "primitive")
        s = "c_" * s
    end
    return s
end
