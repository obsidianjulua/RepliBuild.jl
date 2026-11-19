# RustGlue.jl — the Rust that RepliBuild writes per type, so Julia can hold a
# value that owns something.
#
# A Rust value that owns memory (a `String`, a `Vec`, a `Box`, a record that
# holds one) must be ended by Rust's own drop code for that type, once. Which
# types have such code is the compiler's knowledge alone: a record of one
# integer has it when its author wrote `impl Drop`, and a record with a raw
# pointer has none unless they did. So the build asks the compiler, with one
# generated line per type, in the root crate (`rb_glue.rs`):
#
#     pub unsafe extern "C" fn __rb_drop_7(p: *mut dep::Token,
#         _owns: *const [u8; core::mem::needs_drop::<dep::Token>() as usize]) -> usize { … }
#
# The function runs the type's drop code on the value at `p`. Its own debug
# info then says two things, and both are read back, not assumed:
#
# - what `p` points at, by the compiler's name for it. The line was written
#   from a spelling this file chose; the DWARF says which type the spelling
#   named. A different type is a refusal.
# - the length of the array `_owns` points at, which the compiler computed:
#   1 when dropping the type does anything, 0 when it is plain data.
#
# The hard part is the spelling. DWARF names a type by where it is DEFINED
# (`dep::inner::Token`), and Rust source can only name it by a path that is
# public (`dep::Token`, when the crate re-exports it from a private module).
# Nothing in the library says which paths are public, so the build tries:
# every candidate path is one `use` line of a probe compile, the compiler
# reports the lines that do not resolve, and the first candidate that does is
# used. A type with no path that resolves has no line, and says so.
#
# One row is one line of the file, so a compile error names its row, and a
# row that does not compile is dropped and the rest built again
# (`RustBuild` drives the compiles; nothing here runs a tool).
#
# Design: docs/updates/RUST_FRONT_END.md, Part III.

module RustGlue

import .._parse_json

export Row, Plan, wanted, vec_element, payload_enum

"The generated file of the root crate that holds one line per type."
const FILE = "rb_glue.rs"

"One type's line: its number (`__rb_drop_<n>`), the type's key in the metadata, how the line spells it, and whether it is a `Vec` (which gets two more functions)."
struct Row
    n::Int
    key::String
    spelling::String
    vec::Bool
end

"""
What `rb_glue.rs` was written from: the type keys it was asked for
(`wanted`, sorted), the rows that are in the file, and for each wanted type
with no row, why.
"""
struct Plan
    wanted::Vector{String}
    rows::Vector{Row}
    refused::Dict{String,String}
end
Plan() = Plan(String[], Row[], Dict{String,String}())

drop_symbol(n::Integer) = "__rb_drop_$n"
vec_view_symbol(n::Integer) = "__rb_vec_view_$n"
vec_new_symbol(n::Integer) = "__rb_vec_new_$n"

"The fixed functions of the root crate that build and read a `String`."
const STRING_HELPERS = ("__rb_string_new", "__rb_string_view")

# =============================================================================
# Which types
# =============================================================================

"""
    payload_enum(t) -> Union{Symbol,Nothing}

`:option` for `core::option::Option<T>`, `:result` for
`core::result::Result<T, E>`, by the scope and name the compiler gave the
type; `nothing` for any other.
"""
function payload_enum(t::AbstractDict)::Union{Symbol,Nothing}
    get(t, "kind", "") == "enum" || return nothing
    scope = get(t, "scope", Any[])
    name = String(get(t, "name", ""))
    (scope == ["core", "option"] && startswith(name, "Option<")) && return :option
    (scope == ["core", "result"] && startswith(name, "Result<")) && return :result
    return nothing
end

"""
    vec_element(types, key) -> Union{String,Nothing}

The element type of `alloc::vec::Vec<T, A>`: its first type parameter, as the
DWARF records it. `nothing` for any other type, and for a `Vec` whose
parameters the metadata does not carry.
"""
function vec_element(types::AbstractDict, key::AbstractString)::Union{String,Nothing}
    t = get(types, key, nothing)
    t === nothing && return nothing
    (get(t, "kind", "") == "struct" && get(t, "scope", Any[]) == ["alloc", "vec"] &&
     startswith(String(get(t, "name", "")), "Vec<")) || return nothing
    params = get(t, "params", Any[])
    isempty(params) && return nothing
    e = get(params[1], "type", nothing)
    return e isa AbstractString ? String(e) : nothing
end

"""
    wanted(functions, types) -> Vector{String}

The types the build asks the compiler about, sorted: every record, enum,
union and `Box` that a callable function takes or returns by value, what such
a value is made of where Julia may take it apart (a tuple's fields, the
payloads of an `Option` or a `Result`, a `Vec`'s elements, a `Box`'s target),
and the targets of the references and pointers those functions take, since a
`&mut Vec<f64>` needs a `Vec<f64>` Julia can make.

A scalar, a reference and a fieldless enum are never asked about.
"""
function wanted(functions, types::AbstractDict)::Vector{String}
    out = Set{String}()
    for f in functions
        f["refusal"] === nothing || continue
        for p in f["parameters"]
            String(get(p, "pass", "")) == "ignore" && continue
            _collect!(out, types, get(p, "type", nothing), 0)
        end
        r = f["return"]
        String(get(r, "pass", "")) == "ignore" || _collect!(out, types, get(r, "type", nothing), 0)
    end
    return sort!(collect(out))
end

function _collect!(out::Set{String}, types::AbstractDict, key, depth::Int)
    (key isa AbstractString && depth <= 12) || return
    t = get(types, key, nothing)
    t === nothing && return
    kind = get(t, "kind", "")
    if kind == "pointer"
        pk = get(t, "pointer", "other")
        if pk == "box"
            key in out && return
            push!(out, String(key))
            _collect!(out, types, get(t, "target", nothing), depth + 1)
        elseif pk in ("ref", "mut_ref", "const_ptr", "mut_ptr")
            _collect!(out, types, get(t, "target", nothing), depth + 1)
        end
    elseif kind == "array"
        _collect!(out, types, get(t, "element", nothing), depth + 1)
    elseif kind in ("struct", "enum", "union")
        if !isempty(get(t, "fat", ""))
            # A borrowed slice or trait object points at something unsized,
            # which is not a value. A `Box<[T]>` or `Box<dyn Trait>` owns it.
            (startswith(key, "&") || startswith(key, "*")) || push!(out, String(key))
            return
        end
        fields = get(t, "fields", Any[])
        if get(t, "tuple", false) === true
            foreach(f -> _collect!(out, types, get(f, "type", nothing), depth + 1), fields)
            return
        end
        get(t, "size", nothing) isa Integer || return
        key in out && return
        push!(out, String(key))
        if payload_enum(t) !== nothing
            for v in get(t, "variants", Any[])
                vt = get(types, get(v, "type", ""), nothing)
                vt === nothing && continue
                foreach(f -> _collect!(out, types, get(f, "type", nothing), depth + 1), get(vt, "fields", Any[]))
            end
        end
        _collect!(out, types, vec_element(types, key), depth + 1)
    end
    return
end

# =============================================================================
# Spelling a type
# =============================================================================

# What a path segment cannot be in Rust source of this edition.
const _KEYWORDS = Set(["as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
                       "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
                       "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super",
                       "trait", "true", "type", "unsafe", "use", "where", "while", "abstract", "become",
                       "box", "do", "final", "gen", "macro", "override", "priv", "try", "typeof", "unsized",
                       "virtual", "yield"])

const _PATH = r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+"

"""
    source_name(key) -> String

A type's name as source can write it, before its paths are chosen: the
compiler's name with the default allocator argument cut. `Vec<f64>` is
`alloc::vec::Vec<f64, alloc::alloc::Global>` in DWARF, and `Global` cannot be
named in stable Rust; it is always the last argument, and always the default.
"""
source_name(key::AbstractString)::String = replace(String(key), r",\s*alloc::alloc::Global(?=\s*>)" => "")

"""
    unnameable(key) -> Union{String,Nothing}

Why no Rust source can spell this type, or `nothing` when one might: the
compiler names a closure, an `impl` block or a type local to a function with
braces (`{closure_env#0}`), which is not syntax, and a path through a keyword
is not either. What passes here is tried; the compiler has the last word.
"""
function unnameable(key::AbstractString)::Union{String,Nothing}
    name = source_name(key)
    occursin(r"[{}#]", name) && return "the compiler's name for it (`$key`) has a part with no name in source"
    occursin(r"^[A-Za-z0-9_:<>,;&*()\[\]\s+='\"!.-]+$", name) || return "its name `$key` is not one this spells"
    flat = replace(name, "->" => "")
    for (open, close) in (('<', '>'), ('(', ')'), ('[', ']'))
        depth = 0
        for c in flat
            c == open && (depth += 1)
            c == close && (depth -= 1)
            depth < 0 && return "its name `$key` is not one this spells"
        end
        depth == 0 || return "its name `$key` is not one this spells"
    end
    for p in paths(name), seg in eachsplit(p, "::")
        seg in _KEYWORDS && return "its path `$p` goes through `$seg`, which is a keyword"
    end
    return nothing
end

"Every path of two or more segments in a type's name, once each, in order of appearance."
function paths(name::AbstractString)::Vector{String}
    out = String[]
    for m in eachmatch(_PATH, name)
        p = String(m.match)
        p in out || push!(out, p)
    end
    return out
end

"""
    candidates(path, crate) -> Vector{String}

The paths that may name, from outside its crate, the item defined at `path`,
most likely first: the path itself, then the path with some of its middle
segments left out (a crate re-exports an item of a private module from the
module above, or from its root: `dep::inner::Token` is `dep::Token`), fewest
left out first. For an item of another crate than the wrapped one and the
standard library, the same paths again under the wrapped crate, which may
re-export it. At most 48.
"""
function candidates(path::AbstractString, crate::AbstractString)::Vector{String}
    segs = String[String(s) for s in eachsplit(path, "::")]
    length(segs) >= 2 || return String[String(path)]
    out = String[]
    function add!(head::String, mid::Vector{String}, last::String)
        n = length(mid)
        if n > 5                                   # 2^n paths: the path itself, and its two ends
            for c in (join(vcat(head, mid, last), "::"), head * "::" * last)
                c in out || push!(out, c)
            end
            return
        end
        for keep in n:-1:0, mask in _subsets(n, keep)
            c = join(vcat(head, mid[mask], last), "::")
            c in out || push!(out, c)
        end
    end
    add!(segs[1], segs[2:end-1], segs[end])
    segs[1] in (String(crate), "core", "alloc", "std") || add!(String(crate), segs[1:end-1], segs[end])
    return out[1:min(end, 48)]
end

# The ways to keep `k` of `n` positions, as index lists, earlier positions first.
function _subsets(n::Int, k::Int)::Vector{Vector{Int}}
    k == 0 && return [Int[]]
    k > n && return Vector{Int}[]
    out = Vector{Int}[]
    idx = collect(1:k)
    while true
        push!(out, copy(idx))
        i = k
        while i >= 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in i+1:k
            idx[j] = idx[j-1] + 1
        end
    end
    return out
end

"""
    spell(key, chosen) -> Union{String,Nothing}

The type's spelling with each of its paths replaced by the one chosen for it
(`chosen`: defined path => public path). `nothing` when a path has none.
"""
function spell(key::AbstractString, chosen::AbstractDict)::Union{String,Nothing}
    name = source_name(key)
    all(p -> haskey(chosen, p), paths(name)) || return nothing
    return replace(name, _PATH => p -> chosen[String(p)])
end

# =============================================================================
# The file
# =============================================================================

const _HEADER = "// Generated by RepliBuild: one line per type. Rewritten by the build; an edit here is lost.\n"

"""
    probe_text(candidates) -> (text, line => (path, candidate))

The probe: one `use` line per candidate path. The compiler reports the lines
whose path does not resolve from outside the crate.
"""
function probe_text(cands::AbstractDict)
    io = IOBuffer()
    print(io, _HEADER)
    lines = Dict{Int,Tuple{String,String}}()
    n = count(==('\n'), _HEADER)
    for path in sort!(collect(keys(cands))), c in cands[path]
        n += 1
        println(io, "#[allow(unused_imports)] use $c as _;")
        lines[n] = (String(path), String(c))
    end
    return (String(take!(io)), lines)
end

"""
    glue_text(rows) -> (text, row number => line)

`rb_glue.rs`: one line per row. `__rb_drop_<n>(p, _)` runs the type's drop
code on the value at `p` (0 when done); called with a null pointer it returns
`n` and does nothing, which also keeps every one of these functions different
from the others: a compiler that finds two functions with the same code keeps
one and drops the other's debug info, which is what this file is read for.
The length of the array `_owns` points at is the compiler's answer to "does
dropping this type do anything". A `Vec` gets two more functions, to read
its elements and to build one.
"""
function glue_text(rows::Vector{Row})
    io = IOBuffer()
    print(io, _HEADER)
    lines = Dict{Int,Int}()
    n = count(==('\n'), _HEADER)
    for r in rows
        n += 1
        s = r.spelling
        print(io, "#[no_mangle] pub unsafe extern \"C\" fn $(drop_symbol(r.n))(p: *mut $s, ",
              "_owns: *const [u8; core::mem::needs_drop::<$s>() as usize]) -> usize ",
              "{ if p.is_null() { $(r.n) } else { __rb_drop(p) } }")
        if r.vec
            print(io, " #[no_mangle] pub unsafe extern \"C\" fn $(vec_view_symbol(r.n))(v: *const $s, ",
                  "data: *mut *const u8, len: *mut usize) { __rb_vec_view(v, data, len) }",
                  " #[no_mangle] pub unsafe extern \"C\" fn $(vec_new_symbol(r.n))(data: *const u8, len: usize, ",
                  "out: *mut $s) { __rb_vec_new(data, len, out) }")
        end
        println(io)
        lines[r.n] = n
    end
    return (String(take!(io)), lines)
end

"""
    error_lines(messages) -> (lines, other)

What a failed compile said, from cargo's JSON messages: `lines` maps each
line of `rb_glue.rs` an error names to that error's text, and `other` holds
the errors that name no line of it, as the compiler rendered them.
"""
function error_lines(messages::AbstractString)
    lines = Dict{Int,String}()
    other = String[]
    for line in eachsplit(messages, '\n')
        startswith(line, "{") || continue
        msg = try
            _parse_json(line)
        catch
            continue
        end
        get(msg, "reason", "") == "compiler-message" || continue
        m = get(msg, "message", nothing)
        (m isa AbstractDict && get(m, "level", "") == "error") || continue
        text = String(get(m, "message", ""))
        startswith(text, "aborting due to") && continue
        hit = false
        for s in get(m, "spans", Any[])
            basename(String(get(s, "file_name", ""))) == FILE || continue
            n = get(s, "line_start", nothing)
            n isa Integer || continue
            get!(lines, Int(n), text)
            hit = true
        end
        hit || push!(other, String(strip(String(something(get(m, "rendered", nothing), text)))))
    end
    return (lines, other)
end

"The compiler's errors of a failed build, rendered, from cargo's JSON messages."
function rendered_errors(messages::AbstractString)::String
    out = String[]
    for line in eachsplit(messages, '\n')
        startswith(line, "{") || continue
        msg = try
            _parse_json(line)
        catch
            continue
        end
        get(msg, "reason", "") == "compiler-message" || continue
        m = get(msg, "message", nothing)
        (m isa AbstractDict && get(m, "level", "") == "error") || continue
        r = get(m, "rendered", nothing)
        r isa AbstractString && push!(out, String(strip(r)))
    end
    return join(out, "\n\n")
end

# =============================================================================
# Reading it back
# =============================================================================

"""
    verify(plan, dwarf_functions, types, exported) -> (glue, refused)

What the built library says about each row. `glue[key]` is, for a type whose
line is in the library and names that type:

- `drop`: the function that drops a value of it;
- `owns`: whether dropping one does anything (the compiler's answer);
- `spelling`: how the line named it;
- `vec`: for a `Vec`, the functions `view` and `new`.

`refused[key]` says why a wanted type has none: its plan refused it, its
function is not in the library or has no debug info, or its spelling named
another type.
"""
function verify(plan::Plan, dwarf_functions::AbstractDict, types::AbstractDict, exported)
    glue = Dict{String,Any}()
    refused = Dict{String,String}(plan.refused)
    target(key) = (t = get(types, something(key, ""), nothing);
                   (t !== nothing && get(t, "kind", "") == "pointer") ? get(t, "target", nothing) : nothing)
    for r in plan.rows
        sym = drop_symbol(r.n)
        if !(sym in exported)
            refused[r.key] = "its drop function `$sym` is not in the library"
            continue
        end
        f = get(dwarf_functions, sym, nothing)
        params = f === nothing ? Any[] : get(f, "parameters", Any[])
        if length(params) != 2
            refused[r.key] = "its drop function `$sym` has no debug info"
            continue
        end
        named = target(params[1]["type"])
        if named != r.key
            refused[r.key] = "`$(r.spelling)` names another type" * (named === nothing ? "" : " (`$named`)")
            continue
        end
        owns = get(get(types, something(target(params[2]["type"]), ""), Dict()), "count", nothing)
        if !(owns in (0, 1))
            refused[r.key] = "the compiler's answer to whether it is dropped was not read"
            continue
        end
        entry = Dict{String,Any}("drop" => sym, "owns" => owns == 1, "spelling" => r.spelling)
        if r.vec && vec_view_symbol(r.n) in exported && vec_new_symbol(r.n) in exported
            entry["vec"] = Dict{String,Any}("view" => vec_view_symbol(r.n), "new" => vec_new_symbol(r.n))
        end
        glue[r.key] = entry
    end
    return (glue, refused)
end

end # module RustGlue
