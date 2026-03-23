# =============================================================================
# C++ IDENTIFIER GENERATION
# =============================================================================

"""
    make_cpp_identifier(name::String)::String

Convert C++ symbol to valid Julia identifier.

# Rules
- Remove C++ namespaces (Foo::bar → bar)
- Replace invalid characters with underscore
- Ensure starts with letter or underscore
- Avoid Julia keywords
- Handle operator overloads gracefully
"""
function make_cpp_identifier(name::String)::String
    if isempty(name)
        return ""
    end

    # Strip parameter list from demangled C++ names
    depth = 0
    paren_pos = 0
    for i in eachindex(name)
        c = name[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth -= 1
        elseif c == '(' && depth == 0
            paren_pos = i
            break
        end
    end
    clean = paren_pos > 0 ? name[1:prevind(name, paren_pos)] : name

    # Remove [abi:cxx11] tags
    clean = replace(clean, r"\[abi:[^\]]*\]" => "")

    # Remove C++ namespace
    clean = replace(clean, r"^.*::" => "")

    # Handle C++ operators — longest matches first to avoid partial replacement
    # (e.g. "operator<<" must not match "operator<" first → "op_lt<")
    clean = replace(clean, "operator<<=" => "op_lshift_assign")
    clean = replace(clean, "operator>>=" => "op_rshift_assign")
    clean = replace(clean, "operator<<" => "op_lshift")
    clean = replace(clean, "operator>>" => "op_rshift")
    clean = replace(clean, "operator<=" => "op_le")
    clean = replace(clean, "operator>=" => "op_ge")
    clean = replace(clean, "operator==" => "op_eq")
    clean = replace(clean, "operator!=" => "op_neq")
    clean = replace(clean, "operator+=" => "op_add_assign")
    clean = replace(clean, "operator-=" => "op_sub_assign")
    clean = replace(clean, "operator*=" => "op_mul_assign")
    clean = replace(clean, "operator/=" => "op_div_assign")
    clean = replace(clean, "operator->" => "op_arrow")
    clean = replace(clean, "operator()" => "op_call")
    clean = replace(clean, "operator[]" => "op_getindex")
    clean = replace(clean, "operator+" => "op_add")
    clean = replace(clean, "operator-" => "op_sub")
    clean = replace(clean, "operator*" => "op_mul")
    clean = replace(clean, "operator/" => "op_div")
    clean = replace(clean, "operator<" => "op_lt")
    clean = replace(clean, "operator>" => "op_gt")
    clean = replace(clean, "operator=" => "op_assign")
    clean = replace(clean, "operator!" => "op_not")
    clean = replace(clean, "operator~" => "op_bitnot")
    clean = replace(clean, "operator&" => "op_bitand")
    clean = replace(clean, "operator|" => "op_bitor")
    clean = replace(clean, "operator^" => "op_bitxor")

    # Replace invalid characters with underscore
    clean = replace(clean, r"[^a-zA-Z0-9_!]" => "_")

    # Remove consecutive underscores
    clean = replace(clean, r"_{2,}" => "_")

    # Ensure starts with letter or underscore
    if !isempty(clean) && isdigit(clean[1])
        clean = "_" * clean
    end

    # Avoid Julia keywords
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
