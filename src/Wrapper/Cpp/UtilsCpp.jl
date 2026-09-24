# =============================================================================
# C++ UTILITY FUNCTIONS
# =============================================================================

# Helper: sanitize a C++ type name to a valid Julia struct/type identifier
function _sanitize_cpp_type_name(name::AbstractString)::String
    s = replace(string(name), "::" => "_")
    s = replace(s, "<"  => "_")
    s = replace(s, ">"  => "")
    s = replace(s, ","  => "_")
    s = replace(s, " "  => "_")
    s = replace(s, "-"  => "minus_")
    s = replace(s, "+"  => "plus_")
    s = replace(s, "*"  => "star_")
    # Catch-all: anything still not identifier-legal becomes '_'. The named
    # replacements above are a curated list, and DWARF emits C++ spellings that
    # are not on it — a lambda's type is literally
    #   (lambda at ./src/llama-model-loader.cpp:1538:79)
    # carrying '(', ')', '/', '.' and single ':'. Those reached a struct FIELD
    # unaltered and the emitted module was a **syntax error**, so all 98k lines
    # of the llama.cpp wrapper were dead — not one function callable. The struct
    # NAME survived only because its emitter happened to apply a stricter pass,
    # which is exactly the kind of disagreement a total function removes.
    # Same idiom already used for member names (GeneratorCpp.jl ~979). Runs
    # before the '_+' collapse so a run of substituted characters folds to one.
    s = replace(s, r"[^A-Za-z0-9_]" => "_")
    # Collapse consecutive underscores and trim trailing ones so that every
    # call-site (struct definitions, field types, function parameters, …)
    # produces identical identifiers for the same C++ type.
    s = replace(s, r"_+" => "_")
    # A C++ name that is ALL underscores survives every replacement above as a
    # single "_", which the rstrip below then empties. oneDNN spells the zero
    # enumerator of both `enum class inner_blk_t { _, _4a, … }` and
    # `block_dim_t { _, _A, … }` literally `_` (src/common/tag_traits.hpp), and
    # `_` is a perfectly legal C++ identifier, so DWARF records it faithfully.
    # The empty string then reached an identifier position and the emitted
    # module was a syntax error — `@enum inner_blk_t::Cint begin` followed by
    # ` = 0` — which `_assert_wrapper_parses` refuses, killing a 271k-line
    # wrapper over two enumerators.
    #
    # `_` is NOT a usable repair. Julia parses `@enum T _ = 0` and evaluates it,
    # but all-underscore identifiers are WRITE-ONLY — "their values cannot be
    # used in expressions" — so the enumerator could never be referenced. Here
    # that is value 0, the C++ default (`inner_blk_t{}`), and a member nobody
    # can name is barely better than the dropped member this must not become.
    # Escape it with the same `c_` prefix the keyword and Base-collision
    # branches below use; `c__` reads back, round-trips through `T(0)` and
    # shows up in `instances()`.
    #
    # `_sanitize_c_type_name` (Wrapper/C/UtilsC.jl) guards the empty case and
    # this one did not — the same two-generators-one-guard split as the
    # namespace phantom-`this` bug. Both are fixed; keep them in step.
    #
    # Note the asymmetry that REMAINS on purpose: C returns "_UnknownType" for a
    # genuinely empty name, C++ still returns "". Do not "fix" that to match —
    # callers here use empty as a SENTINEL. GeneratorCpp.jl:2390 reads
    # `isempty(inner_safe) ? "Ptr{Cvoid}" : "$wrapper_kw{$inner_safe}"`, so a
    # non-empty placeholder would emit `Ptr{_UnknownType}` where `Ptr{Cvoid}`
    # is meant; GeneratorCpp.jl:1553 gates on it too. All-underscore is a
    # distinct case from empty, which is why it is repaired here and empty is
    # left alone.
    was_all_underscore = !isempty(s) && all(==('_'), s)
    s = String(rstrip(s, '_'))
    was_all_underscore && (s = "c__")
    if !isempty(s) && isdigit(s[1])
        s = "_" * s
    end
    if s in ("for", "if", "else", "while", "function", "struct", "end", "module", "using", "import", "export", "return", "continue", "break", "try", "catch", "finally", "macro", "quote", "let", "local", "global", "const", "do", "baremodule", "true", "false", "abstract", "type", "mutable", "primitive")
        s = "c_" * s
    end
    # Capitalized Base/Core bindings are not keywords, so they survive the check
    # above, but a C++ type emitted under one of these bare names (e.g. a nested
    # `enum Type`, a `Ref`/`Ptr`/`Vector` class) shadows the builtin and breaks
    # every `::Type{…}`/`Ptr{…}` annotation in the generated module. Rename them
    # the same way so the emitted identifier never collides with Base.
    if s in ("Type", "Ref", "Ptr", "Vector", "Array", "Any", "Val", "Module",
             "Function", "Tuple", "Union", "NTuple", "Nothing", "Some", "Pair",
             "Dict", "Set", "Enum", "String", "Symbol", "Expr", "Base", "Core", "Main")
        s = "c_" * s
    end
    return s
end

"""
    _is_julia_type_spelling(t) -> Bool

True when `t` is already written as a Julia type and must be passed through
untouched (`Int32`, `Ptr{Cvoid}`, `NTuple{4, UInt8}`), false when it is still a
raw C++ spelling that needs sanitizing (`initializer_list<range_nfd>`,
`std::vector<int>`).

The distinction matters because the two cases need opposite treatment: running
`_sanitize_cpp_type_name` over `Ptr{Cvoid}` would corrupt it into `Ptr_Cvoid`,
while NOT running it over a C++ template spelling emits `<`/`>` into an
identifier position and produces a syntax error. Testing the character set is
enough — Julia type expressions use only word characters, braces, commas and
spaces, and every C++ construct that breaks the parser (`<`, `>`, `(`, `:`,
`*`, `/`) is outside that set.
"""
function _is_julia_type_spelling(t::AbstractString)::Bool
    isempty(t) && return false
    # `Name` or `Name{...}` — spaces are legal only INSIDE the braces. Allowing
    # them anywhere let `unsigned int` through, which is two juxtaposed
    # identifiers and just as much a syntax error in a `::` position as `<` is.
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*(\{[A-Za-z0-9_{}, ]*\})?$", t) || return false
    return count(==('{'), t) == count(==('}'), t)
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
