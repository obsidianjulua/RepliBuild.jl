# RustABI.jl — how each function of a Rust crate is actually called.
#
# rustc's own calling convention (`extern "Rust"`) is not C's and is not
# written down anywhere a tool may rely on: a `&str` is two arguments, an
# `Option<i32>` comes back in two registers, a 12-byte record goes by pointer
# to a copy. RepliBuild does not predict it. The compiler wrote what it did
# for THIS build on each function's `define` line of the IR it emitted, and
# this module reads that line (`definitions`), the way
# `Compiler._byval_definitions` reads clang's.
#
# A `define` line says how a call is made and not what an argument means: a
# `ptr` is a borrow, a pointer value, or a copy the callee owns. DWARF says
# that, one formal parameter per Rust parameter. `join_function` matches the
# two: each DWARF parameter takes 0, 1 or 2 IR parameters, by its type's
# shape. A function whose IR parameters are not all accounted for, in exactly
# one way, is REFUSED and its reason recorded. Nothing here guesses.
#
# Which symbols are the crate's own API is `surface`: exported, defined by the
# crate's IR, and rooted in the crate by its v0 mangled path (`v0_root`), not
# by the crate's name appearing somewhere in the symbol.
#
# Toolchain-free: text in, tables out. Design: docs/updates/RUST_FRONT_END.md.

module RustABI

import ..Compiler

export v0_root, definitions, aliases, alias_target, surface, join_function, shape

# =============================================================================
# v0 symbol names: whose item is this
# =============================================================================

"""
What a v0 symbol's path says about where its item lives.

- `crate`, `disambiguator`: the crate at the root of the path. For a method
  it is the crate of the `impl` block; for an instantiation of a generic
  function, the crate that DEFINES the generic, not the one that instantiated
  it (`core::ptr::drop_in_place::<String>` is rooted in `core`); for a
  trait's provided method, the trait's crate.
- `generic`: the path has generic arguments on its spine.
- `namespace`: of the outermost path component: `v` for a function or a
  static, `C` for a closure, `S` for a shim.
"""
struct SymbolRoot
    crate::String
    disambiguator::String
    generic::Bool
    namespace::Char
end

function _base62(s::AbstractString)::Union{Int,Nothing}
    isempty(s) && return 0
    v = 0
    for c in s
        d = '0' <= c <= '9' ? c - '0' :
            'a' <= c <= 'z' ? c - 'a' + 10 :
            'A' <= c <= 'Z' ? c - 'A' + 36 : return nothing
        v = v * 62 + d
        v > typemax(Int32) && return nothing
    end
    return v + 1
end

# --- Skipping over the parts of a v0 name that are not on the way to its root.
# Each `_skip_*` takes the index of the first byte of the thing and returns the
# index after it, or 0 when the text is not that thing. The grammar is the
# one in the Rust reference's "v0 symbol format"; a back reference is stepped
# over, not followed.

_at(s::String, i::Int) = (1 <= i <= ncodeunits(s)) ? Char(codeunit(s, i)) : '\0'

function _skip_base62(s::String, i::Int)::Int       # {<0-9a-zA-Z>} "_"
    while isletter(_at(s, i)) || isdigit(_at(s, i))
        i += 1
    end
    return _at(s, i) == '_' ? i + 1 : 0
end

function _skip_ident(s::String, i::Int)::Int        # [disambiguator] ["u"] <decimal> ["_"] <bytes>
    if _at(s, i) == 's'
        i = _skip_base62(s, i + 1)
        i == 0 && return 0
    end
    _at(s, i) == 'u' && (i += 1)
    j = i
    while isdigit(_at(s, j))
        j += 1
    end
    j == i && return 0
    len = parse(Int, s[i:j-1])
    _at(s, j) == '_' && (j += 1)
    return j + len - 1 <= ncodeunits(s) ? j + len : 0
end

function _skip_path(s::String, i::Int, depth::Int = 0)::Int
    depth > 64 && return 0
    c = _at(s, i)
    if c == 'C'
        return _skip_ident(s, i + 1)
    elseif c == 'N'
        i = _skip_path(s, i + 2, depth + 1)
        return i == 0 ? 0 : _skip_ident(s, i)
    elseif c == 'M' || c == 'X'
        i += 1
        if _at(s, i) == 's'
            i = _skip_base62(s, i + 1)
            i == 0 && return 0
        end
        i = _skip_path(s, i, depth + 1)
        i == 0 && return 0
        i = _skip_type(s, i, depth + 1)
        (i == 0 || c == 'M') && return i
        return _skip_path(s, i, depth + 1)
    elseif c == 'Y'
        i = _skip_type(s, i + 1, depth + 1)
        return i == 0 ? 0 : _skip_path(s, i, depth + 1)
    elseif c == 'I'
        i = _skip_path(s, i + 1, depth + 1)
        while i != 0 && _at(s, i) != 'E'
            i = _skip_generic_arg(s, i, depth + 1)
        end
        return i == 0 ? 0 : i + 1
    elseif c == 'B'
        return _skip_base62(s, i + 1)
    end
    return 0
end

function _skip_generic_arg(s::String, i::Int, depth::Int)::Int
    c = _at(s, i)
    c == 'L' && return _skip_base62(s, i + 1)
    c == 'K' && return _skip_const(s, i + 1, depth)
    return _skip_type(s, i, depth)
end

function _skip_const(s::String, i::Int, depth::Int)::Int
    c = _at(s, i)
    c == 'p' && return i + 1
    c == 'B' && return _skip_base62(s, i + 1)
    i = _skip_type(s, i, depth + 1)
    i == 0 && return 0
    _at(s, i) == 'n' && (i += 1)
    while isxdigit(_at(s, i))
        i += 1
    end
    return _at(s, i) == '_' ? i + 1 : 0
end

const _V0_BASIC_TYPES = Set("abcdefhijlmnopstuvxyz")

function _skip_type(s::String, i::Int, depth::Int = 0)::Int
    depth > 64 && return 0
    c = _at(s, i)
    c in _V0_BASIC_TYPES && return i + 1
    if c == 'A'                                     # [T; N]
        i = _skip_type(s, i + 1, depth + 1)
        return i == 0 ? 0 : _skip_const(s, i, depth + 1)
    elseif c == 'S' || c == 'P' || c == 'O'         # [T], *const T, *mut T
        return _skip_type(s, i + 1, depth + 1)
    elseif c == 'R' || c == 'Q'                     # &T, &mut T
        i += 1
        if _at(s, i) == 'L'
            i = _skip_base62(s, i + 1)
            i == 0 && return 0
        end
        return _skip_type(s, i, depth + 1)
    elseif c == 'T'                                 # (T, U)
        i += 1
        while i != 0 && _at(s, i) != 'E'
            i = _skip_type(s, i, depth + 1)
        end
        return i == 0 ? 0 : i + 1
    elseif c == 'F'                                 # fn(T) -> U
        i += 1
        if _at(s, i) == 'G'
            i = _skip_base62(s, i + 1)
            i == 0 && return 0
        end
        _at(s, i) == 'U' && (i += 1)
        if _at(s, i) == 'K'
            i = _at(s, i + 1) == 'C' ? i + 2 : _skip_ident(s, i + 1)
            i == 0 && return 0
        end
        while i != 0 && _at(s, i) != 'E'
            i = _skip_type(s, i, depth + 1)
        end
        return i == 0 ? 0 : _skip_type(s, i + 1, depth + 1)
    elseif c == 'D'                                 # dyn Trait + 'a
        i += 1
        if _at(s, i) == 'G'
            i = _skip_base62(s, i + 1)
            i == 0 && return 0
        end
        while i != 0 && _at(s, i) != 'E'
            i = _skip_path(s, i, depth + 1)
            while i != 0 && _at(s, i) == 'p'
                i = _skip_ident(s, i + 1)
                i == 0 || (i = _skip_type(s, i, depth + 1))
            end
        end
        i == 0 && return 0
        return _at(s, i + 1) == 'L' ? _skip_base62(s, i + 2) : 0
    end
    return _skip_path(s, i, depth + 1)              # a named type, or a back reference
end

"""
    v0_root(symbol) -> Union{SymbolRoot,Nothing}

Walk a v0-mangled Rust symbol (`_R…`) down its path to the crate at its root:
through a nested path (`N`), an inherent or trait `impl` (`M`, `X`, whose
path comes before its type), generic arguments (`I`, whose path comes
first), a back reference (`B`), to the crate (`C`). For a trait's own item
used on some type (`Y`: `<T as Trait>::provided_method`) the root is the
TRAIT's, since that is where the body is written; the type is stepped over.

`nothing` for a name that is not v0 (a legacy `_ZN…`), or one whose grammar
this does not read.
"""
function v0_root(symbol::AbstractString)::Union{SymbolRoot,Nothing}
    startswith(symbol, "_R") || return nothing
    s = String(symbol)
    isascii(s) || return nothing
    i = 3
    while isdigit(_at(s, i))            # an encoding version, should one ever exist
        i += 1
    end
    generic = false
    namespace = '\0'
    for _ in 1:64
        c = _at(s, i)
        if c == 'C'
            i += 1
            dis = ""
            if _at(s, i) == 's'
                j = _skip_base62(s, i + 1)
                j == 0 && return nothing
                dis = s[i+1:j-2]
                i = j
            end
            _at(s, i) == 'u' && (i += 1)
            j = i
            while isdigit(_at(s, j))
                j += 1
            end
            j == i && return nothing
            len = parse(Int, s[i:j-1])
            _at(s, j) == '_' && (j += 1)
            j + len - 1 <= ncodeunits(s) || return nothing
            return SymbolRoot(s[j:j+len-1], dis, generic, namespace)
        elseif c == 'N'
            namespace == '\0' && (namespace = _at(s, i + 1))
            i += 2
        elseif c == 'M' || c == 'X'
            i += 1
            if _at(s, i) == 's'
                i = _skip_base62(s, i + 1)
                i == 0 && return nothing
            end
        elseif c == 'Y'
            i = _skip_type(s, i + 1)
            i == 0 && return nothing
        elseif c == 'I'
            generic = true
            i += 1
        elseif c == 'B'
            j = _skip_base62(s, i + 1)
            j == 0 && return nothing
            v = _base62(s[i+1:j-2])
            v === nothing && return nothing
            i = 3 + v
        else
            return nothing
        end
    end
    return nothing
end

# =============================================================================
# `define` lines
# =============================================================================

"One parameter of a `define` line: its LLVM type, every attribute as written, and its value name."
struct IRParam
    type::String
    attrs::Vector{String}
    name::String
end

"""
One function definition, as its `define` line states it.

- `cc`: the calling convention when the line names one (`fastcc`); `""` is
  the default, which is the only one a call from outside can make.
- `nounwind`: the definition, or its attribute group, says it cannot unwind.
- `personality`: the unwinding personality it names, or `""`.
"""
struct IRDefinition
    symbol::String
    linkage::Vector{String}
    cc::String
    ret_type::String
    ret_attrs::Vector{String}
    params::Vector{IRParam}
    variadic::Bool
    nounwind::Bool
    personality::String
end

const _GLOBAL_NAME = r"@(\"(?:[^\"\\]|\\.)*\"|[-a-zA-Z$._][-a-zA-Z$._0-9]*)\("
_unquote(n::AbstractString) = startswith(n, "\"") ?
    replace(n[2:end-1], r"\\([0-9A-Fa-f]{2})" => h -> string(Char(parse(UInt8, h[2:3], base = 16)))) : String(n)

const _LINKAGE_WORDS = Set(["private", "internal", "available_externally", "linkonce", "weak",
    "common", "appending", "extern_weak", "linkonce_odr", "weak_odr", "external",
    "dso_local", "dso_preemptable", "default", "hidden", "protected", "dllimport", "dllexport",
    "unnamed_addr", "local_unnamed_addr"])

# Split on the whitespace outside brackets: `{ i32, double }`, `[24 x i8]`,
# `sret([24 x i8])` and `range(i64 0, 10)` are one token each.
function _tokens(s::AbstractString)::Vector{String}
    out = String[]
    buf = IOBuffer()
    pending = false
    depth = 0
    for c in s
        if c in ('(', '[', '{', '<')
            depth += 1
        elseif c in (')', ']', '}', '>')
            depth -= 1
        end
        if isspace(c) && depth == 0
            pending && push!(out, String(take!(buf)))
            pending = false
        else
            write(buf, c)
            pending = true
        end
    end
    pending && push!(out, String(take!(buf)))
    return out
end

# `align 8` is two tokens and one attribute.
function _attrs(tokens::Vector{String})::Vector{String}
    out = String[]
    i = 1
    while i <= length(tokens)
        if tokens[i] == "align" && i < length(tokens)
            push!(out, "align " * tokens[i+1])
            i += 2
        else
            push!(out, tokens[i])
            i += 1
        end
    end
    return out
end

function _parse_param(text::AbstractString)::Union{IRParam,Nothing}
    toks = _tokens(strip(text))
    isempty(toks) && return nothing
    name = (length(toks) > 1 && startswith(toks[end], "%")) ? pop!(toks) : ""
    return IRParam(toks[1], _attrs(toks[2:end]), name)
end

"""
    definitions(ir_files) -> Dict{String,IRDefinition}

Every function DEFINED in `ir_files`, by symbol: its return type and
attributes, its parameters in order (type, attributes, name), its calling
convention, and whether it can unwind. A function the IR only declares is
absent.

`nounwind` is `Compiler._nounwind_definitions`' answer, which resolves an
attribute group (`#N`) in the file that defines it.
"""
function definitions(ir_files)::Dict{String,IRDefinition}
    out = Dict{String,IRDefinition}()
    for f in ir_files
        isfile(f) || continue
        nounwind = Compiler._nounwind_definitions([f])
        for line in eachline(f)
            startswith(line, "define ") || continue
            d = _parse_define(line)
            d === nothing && continue
            out[d.symbol] = IRDefinition(d.symbol, d.linkage, d.cc, d.ret_type, d.ret_attrs, d.params,
                                         d.variadic, d.symbol in nounwind, d.personality)
        end
    end
    return out
end

"""
    aliases(ir_files) -> Dict{String,String}

Each symbol the IR defines as an ALIAS, and the symbol it stands for.

Above opt-level 1 LLVM merges functions whose code is the same, and keeps one
of them as an alias of the other (`@c_add = unnamed_addr alias i32 (i32, i32),
ptr @…3add`). The alias is a symbol of the library like any other, has no
`define` line, and may have no DIE either. `surface` counts it, and its call
is its target's.
"""
function aliases(ir_files)::Dict{String,String}
    name = "(\"(?:[^\"\\\\]|\\\\.)*\"|[-a-zA-Z\$._][-a-zA-Z\$._0-9]*)"
    pattern = Regex("^@" * name * " = .*\\balias\\b.*, ptr @" * name * "\\s*\$")
    out = Dict{String,String}()
    for f in ir_files
        isfile(f) || continue
        for line in eachline(f)
            startswith(line, "@") || continue
            m = match(pattern, line)
            m === nothing || (out[_unquote(m.captures[1])] = _unquote(m.captures[2]))
        end
    end
    return out
end

"The definition an alias ends at, through other aliases, or `nothing`."
function alias_target(symbol::AbstractString, defs::AbstractDict, aliases::AbstractDict)::Union{String,Nothing}
    s = String(symbol)
    for _ in 1:16
        haskey(defs, s) && return s
        haskey(aliases, s) || return nothing
        s = aliases[s]
    end
    return nothing
end

function _parse_define(line::AbstractString)::Union{IRDefinition,Nothing}
    m = match(_GLOBAL_NAME, line)
    m === nothing && return nothing
    symbol = _unquote(m.captures[1])
    # The parameter list: from after `@name(` to its matching paren.
    i = m.offset + ncodeunits(m.match)
    start = i
    depth = 1
    while i <= ncodeunits(line) && depth > 0
        c = codeunit(line, i)
        c == UInt8('(') && (depth += 1)
        c == UInt8(')') && (depth -= 1)
        i += 1
    end
    depth == 0 || return nothing
    params_text = line[start:i-2]
    tail = line[i:end]

    head = _tokens(line[ncodeunits("define ")+1:m.offset-1])
    isempty(head) && return nothing
    ret_type = pop!(head)
    linkage = String[]
    cc = ""
    rest = String[]
    k = 1
    while k <= length(head)
        t = head[k]
        if t in _LINKAGE_WORDS
            push!(linkage, t)
        elseif t == "cc" && k < length(head) && all(isdigit, head[k+1])
            cc = "cc " * head[k+1]
            k += 1
        elseif endswith(t, "cc") && all(c -> islowercase(c) || c == '_', t)
            t == "ccc" || (cc = t)
        else
            push!(rest, t)
        end
        k += 1
    end

    params = IRParam[]
    variadic = false
    if !isempty(strip(params_text))
        for p in Compiler._ir_split_params(params_text)
            if strip(p) == "..."
                variadic = true
                continue
            end
            ip = _parse_param(p)
            ip === nothing && return nothing
            push!(params, ip)
        end
    end
    pm = match(r"\bpersonality ptr @(\"(?:[^\"\\]|\\.)*\"|[-a-zA-Z$._][-a-zA-Z$._0-9]*)", tail)
    return IRDefinition(symbol, linkage, cc, ret_type, _attrs(rest), params, variadic, false,
                        pm === nothing ? "" : _unquote(pm.captures[1]))
end

# =============================================================================
# The surface
# =============================================================================

"""
    surface(defs, exported, crate; aliases = Dict()) -> (symbols, skipped, foreign)

The crate's own callable functions: defined in its IR (`defs`, or an alias of
one in `aliases`), in the library's dynamic symbol table (`exported`), and either a function path
(`namespace` `v`) rooted in `crate` by `v0_root`, or a name rustc did not
mangle (a `#[no_mangle]` item of the crate, whose IR this is). `symbols` is
sorted.

`skipped` names, with a reason, each exported definition whose path this
could not read, so a function left out of the wrapper is never left out
silently. An exported definition rooted in ANOTHER crate (the standard
library's generics, instantiated here) is not the crate's API and is only
counted, in `foreign`.
"""
function surface(defs::AbstractDict{String,IRDefinition}, exported::AbstractSet{<:AbstractString},
                 crate::AbstractString; aliases::AbstractDict = Dict{String,String}())
    symbols = String[]
    skipped = Dict{String,Any}[]
    foreign = 0
    candidates = collect(keys(defs))
    for a in keys(aliases)
        (haskey(defs, a) || alias_target(a, defs, aliases) === nothing) || push!(candidates, String(a))
    end
    for sym in sort!(candidates)
        sym in exported || continue
        root = v0_root(sym)
        if !startswith(sym, "_R")
            # A name rustc did not mangle is a `#[no_mangle]` item, and the
            # IR is the crate's own: it is the crate's function.
            push!(symbols, sym)
        elseif root === nothing
            push!(skipped, Dict{String,Any}("symbol" => sym, "reason" => "its symbol name could not be read"))
        elseif root.crate != crate
            foreign += 1
        elseif root.namespace != 'v'
            push!(skipped, Dict{String,Any}("symbol" => sym, "reason" => "not a function path (a closure or a shim)"))
        else
            push!(symbols, sym)
        end
    end
    return (symbols, skipped, foreign)
end

# =============================================================================
# Type shapes
# =============================================================================

"One scalar inside a type: where it is, how wide, and what kind (`:int`, `:float`, `:ptr`, `:bool`, `:mixed`)."
struct Leaf
    offset::Int
    size::Int
    class::Symbol
end

"""
What the join needs to know about a type.

- `size`: bytes, or -1 when the DWARF does not settle it.
- `kind`: the description's kind (`base`, `pointer`, `struct`, `enum`, …).
- `leaves`: its scalars at their offsets, or a reason there is no such list:
  `:unknown` (a layout this cannot read), `:many` (more than the join could
  use) or `:overlap` (variants that put different scalars over each other).
- `fat`: a fat pointer, which rustc always passes as its two words.
"""
struct Shape
    size::Int
    kind::String
    leaves::Union{Vector{Leaf},Symbol}
    fat::Bool
end

const _MAX_LEAVES = 8
const POINTER_BYTES = Sys.WORD_SIZE ÷ 8

"`shape(types, key)`: the `Shape` of the type `key` names in the table `RustDWARF.describe_functions` built."
function shape(types::AbstractDict, key::Union{AbstractString,Nothing})::Shape
    key === nothing && return Shape(0, "unit", Leaf[], false)
    t = get(types, key, nothing)
    t === nothing && return Shape(-1, "unknown", :unknown, false)
    kind = String(get(t, "kind", "unknown"))
    size = get(t, "size", nothing)
    get(t, "conflict", false) === true && return Shape(-1, kind, :unknown, false)
    kind == "pointer" && return Shape(something(size, POINTER_BYTES), kind, [Leaf(0, something(size, POINTER_BYTES), :ptr)], false)
    size isa Integer || return Shape(-1, kind, :unknown, false)
    return Shape(Int(size), kind, _leaves(types, String(key), 0), !isempty(get(t, "fat", "")))
end

function _shift(ls::Vector{Leaf}, by::Int)
    by == 0 && return ls
    return Leaf[Leaf(l.offset + by, l.size, l.class) for l in ls]
end

# Scalars that different variants put at the same place are one scalar; ones
# that only partly cover each other have no single reading.
function _merge(ls::Vector{Leaf})::Union{Vector{Leaf},Symbol}
    sort!(ls; by = l -> (l.offset, l.size, String(l.class)))
    out = Leaf[]
    for l in ls
        if !isempty(out) && out[end].offset == l.offset && out[end].size == l.size
            if out[end].class !== l.class
                # A float under an integer is still one scalar of that width,
                # but no register class is common to both.
                out[end] = Leaf(l.offset, l.size, (out[end].class === :float || l.class === :float ||
                                                   out[end].class === :mixedf) ? :mixedf : :mixed)
            end
        elseif !isempty(out) && l.offset < out[end].offset + out[end].size
            return :overlap
        else
            push!(out, l)
        end
    end
    return out
end

function _leaves(types::AbstractDict, key::String, depth::Int)::Union{Vector{Leaf},Symbol}
    depth > 32 && return :unknown
    t = get(types, key, nothing)
    t === nothing && return :unknown
    get(t, "conflict", false) === true && return :unknown
    kind = get(t, "kind", "unknown")
    size = get(t, "size", nothing)
    if kind == "pointer"
        return [Leaf(0, something(size, POINTER_BYTES), :ptr)]
    elseif kind == "base"
        size isa Integer || return :unknown
        size == 0 && return Leaf[]
        enc = get(t, "encoding", "other")
        class = enc == "float" ? :float : enc == "boolean" ? :bool :
                enc in ("signed", "unsigned", "signed_char", "unsigned_char", "utf", "address") ? :int : nothing
        class === nothing && return :unknown
        return [Leaf(0, Int(size), class)]
    elseif kind == "c_enum"
        size isa Integer || return :unknown
        return size == 0 ? Leaf[] : [Leaf(0, Int(size), :int)]
    elseif kind == "array"
        size isa Integer || return :unknown
        size == 0 && return Leaf[]
        count = get(t, "count", nothing)
        elem = get(t, "element", nothing)
        (count isa Integer && elem isa AbstractString) || return :unknown
        el = _leaves(types, String(elem), depth + 1)
        el isa Symbol && return el
        (isempty(el) || count == 0) && return Leaf[]
        esize = get(get(types, elem, Dict()), "size", nothing)
        esize isa Integer || return :unknown
        count * length(el) > _MAX_LEAVES && return :many
        out = Leaf[]
        for i in 0:count-1
            append!(out, _shift(el, i * Int(esize)))
        end
        return out
    elseif kind in ("struct", "union", "enum")
        size isa Integer || return :unknown
        size == 0 && return Leaf[]
        out = Leaf[]
        function add!(type_key, offset)
            (type_key isa AbstractString && offset isa Integer) || return :unknown
            sub = _leaves(types, String(type_key), depth + 1)
            sub isa Symbol && return sub
            append!(out, _shift(sub, Int(offset)))
            return length(out) > 4 * _MAX_LEAVES ? :many : nothing
        end
        for f in get(t, "fields", Any[])
            r = add!(get(f, "type", nothing), get(f, "offset", nothing))
            r === nothing || return r
        end
        if kind == "enum"
            d = get(t, "discriminant", nothing)
            if d !== nothing
                r = add!(get(d, "type", nothing), get(d, "offset", nothing))
                r === nothing || return r
            end
            for v in get(t, "variants", Any[])
                haskey(v, "type") || continue
                r = add!(get(v, "type", nothing), something(get(v, "offset", 0), 0))
                r === nothing || return r
            end
        end
        merged = _merge(out)
        merged isa Symbol && return merged
        return length(merged) > _MAX_LEAVES ? :many : merged
    end
    return :unknown
end

# =============================================================================
# The join
# =============================================================================

"The bytes an LLVM scalar type holds, or `nothing` for a type that is not one scalar."
function _ir_scalar_bytes(t::AbstractString)::Union{Int,Nothing}
    t == "ptr" && return POINTER_BYTES
    t == "double" && return 8
    t == "float" && return 4
    t == "half" && return 2
    t == "fp128" && return 16
    t == "i1" && return 1
    m = match(r"^i(\d+)$", t)
    m === nothing && return nothing
    bits = parse(Int, m.captures[1])
    return bits % 8 == 0 ? bits ÷ 8 : nothing
end

function _scalar_matches(leaf::Leaf, ir::AbstractString)::Bool
    leaf.class === :ptr   && return ir == "ptr"
    leaf.class === :bool  && return ir == "i1"
    leaf.class === :int   && return ir == "i$(8 * leaf.size)"
    leaf.class === :float && return (leaf.size == 4 && ir == "float") || (leaf.size == 8 && ir == "double") ||
                                    (leaf.size == 2 && ir == "half") || (leaf.size == 16 && ir == "fp128")
    # An integer in one variant and a pointer in another: rustc passes the word.
    leaf.class === :mixed && return ir == "i$(8 * leaf.size)" || (leaf.size == POINTER_BYTES && ir == "ptr")
    return false
end

const _AGGREGATE_KINDS = ("struct", "union", "enum", "array")

_attr_number(attrs::Vector{String}, name::String) =
    (for a in attrs
         m = match(Regex("^" * name * "\\((\\d+)\\)\$"), a)
         m === nothing || return parse(Int, m.captures[1])
     end; nothing)

"""
    _options(shape, ir, i) -> Vector{Tuple{String,Int}}

The ways a value of `shape` can be the IR parameters starting at `ir[i]`, as
`(pass, how many it takes)`:

- `ignore`, 0: a zero-sized value is not passed.
- `direct`, 1: one scalar of the value's whole size.
- `pair`, 2: two scalars (a fat pointer, a two-field record, `Option<i32>`).
- `cast`, 1: an aggregate of a word or less, as one integer of its size.
- `indirect`, 1: a larger aggregate, by pointer to a copy the caller made.

Every option is checked against the IR types it would take, so a shape rustc
lays out differently than its scalars suggest fails to match instead of
matching wrongly.
"""
function _options(sh::Shape, ir::Vector{IRParam}, i::Int)::Vector{Tuple{String,Int}}
    sh.size == 0 && return [("ignore", 0)]
    sh.size < 0 && return Tuple{String,Int}[]
    n = length(ir)
    out = Tuple{String,Int}[]
    L = sh.leaves
    if L isa Vector{Leaf} && sh.kind != "array"
        if length(L) == 1 && L[1].offset == 0 && L[1].size == sh.size && i <= n &&
           _scalar_matches(L[1], ir[i].type)
            push!(out, ("direct", 1))
        end
        if length(L) == 2 && L[1].offset == 0 && L[2].offset >= L[1].size &&
           L[2].offset + L[2].size <= sh.size && i + 1 <= n &&
           _scalar_matches(L[1], ir[i].type) && _scalar_matches(L[2], ir[i+1].type)
            push!(out, ("pair", 2))
        end
    end
    if sh.kind in _AGGREGATE_KINDS && !sh.fat && i <= n
        if sh.size <= POINTER_BYTES
            ir[i].type == "i$(8 * sh.size)" && !any(o -> o[1] == "direct", out) && push!(out, ("cast", 1))
        elseif ir[i].type == "ptr"
            d = _attr_number(ir[i].attrs, "dereferenceable")
            (d === nothing || d == sh.size) && push!(out, ("indirect", 1))
        end
    end
    return out
end

function _solve(shapes::Vector{Shape}, ir::Vector{IRParam}, first::Int)
    found = Vector{Vector{Tuple{String,Int}}}()
    cur = Tuple{String,Int}[]
    function go(k::Int, i::Int)
        length(found) >= 2 && return
        if k > length(shapes)
            i == length(ir) + 1 && push!(found, copy(cur))
            return
        end
        for o in _options(shapes[k], ir, i)
            push!(cur, o)
            go(k + 1, i + o[2])
            pop!(cur)
        end
    end
    go(1, first)
    return found
end

_sret_bytes(p::IRParam) =
    (for a in p.attrs
         startswith(a, "sret(") || continue
         m = match(r"^sret\(\[(\d+) x i8\]\)$", a)
         return m === nothing ? -1 : parse(Int, m.captures[1])
     end; nothing)

function _brace_elements(t::AbstractString)::Vector{String}
    (startswith(t, "{") && endswith(t, "}")) || return String[]
    return String[String(strip(x)) for x in Compiler._ir_split_params(t[2:end-1])]
end

"""
    join_function(symbol, def, dwarf, types) -> Dict

One function's row of the metadata: what DWARF says it is (`name`, `scope`,
`owner`, each parameter's name and type, the return type), what its `define`
line says (`ir`), and how the two fit: each parameter's `pass` and the IR
parameters it takes (`ir`, 1-based), and the return's.

`refusal` is `nothing` for a function whose call is fully accounted for, and
otherwise the reason it is not: no DIE, a calling convention only the crate
can use, a type whose layout is not known, IR parameters left over (a
`#[track_caller]` function takes one DWARF does not list), or more than one
way to match. A refused function has no `pass` facts and must not be called.
"""
function join_function(symbol::AbstractString, def::IRDefinition, dwarf::Union{AbstractDict,Nothing},
                       types::AbstractDict)::Dict{String,Any}
    root = v0_root(symbol)
    row = Dict{String,Any}(
        "symbol" => String(symbol),
        "generic" => root !== nothing && root.generic,
        # A `#[no_mangle]` item: bound by its own name, usually `extern "C"`.
        "no_mangle" => !startswith(symbol, "_R"),
        "nounwind" => def.nounwind,
        "ir" => Dict{String,Any}(
            "return" => Dict{String,Any}("type" => def.ret_type, "attrs" => def.ret_attrs),
            "parameters" => Any[Dict{String,Any}("type" => p.type, "attrs" => p.attrs, "name" => p.name)
                                for p in def.params],
            "cc" => def.cc,
            "personality" => def.personality),
        "refusal" => nothing)
    refuse(reason) = (row["refusal"] = reason; row)

    if dwarf === nothing
        row["name"] = ""
        row["scope"] = String[]
        row["owner"] = nothing
        row["parameters"] = Any[]
        row["return"] = Dict{String,Any}("type" => nothing)
        return refuse("no debug info describes this function")
    end
    row["name"] = dwarf["name"]
    row["scope"] = dwarf["scope"]
    row["owner"] = dwarf["owner"]
    params = Any[Dict{String,Any}("name" => p["name"], "type" => p["type"]) for p in dwarf["parameters"]]
    row["parameters"] = params
    ret = Dict{String,Any}("type" => dwarf["return"])
    row["return"] = ret

    haskey(dwarf, "dwarf_note") && return refuse("its debug info disagrees with itself: " * dwarf["dwarf_note"])
    isempty(def.cc) || return refuse("its calling convention is `$(def.cc)`, which only the crate's own code can use")
    def.variadic && return refuse("it is variadic")

    # The return.
    rshape = shape(types, dwarf["return"])
    first = 1
    sret = isempty(def.params) ? nothing : _sret_bytes(def.params[1])
    if sret !== nothing
        def.ret_type == "void" || return refuse("it has a hidden return pointer and returns `$(def.ret_type)` too")
        rshape.size > 0 || return refuse("it returns through a hidden pointer a type whose size is not known")
        sret == rshape.size ||
            return refuse("its hidden return pointer is for $(sret < 0 ? "a type this does not size" : "$sret bytes") " *
                          "and the return type is $(rshape.size) bytes")
        ret["pass"] = "indirect"
        ret["ir"] = [1]
        first = 2
    elseif def.ret_type == "void"
        rshape.size == 0 || return refuse("it returns nothing in the IR, and its return type is " *
                                          (rshape.size < 0 ? "one whose layout is not known" : "$(rshape.size) bytes"))
        ret["pass"] = "ignore"
    else
        rshape.size < 0 && return refuse("the layout of its return type is not known")
        elems = _brace_elements(def.ret_type)
        L = rshape.leaves
        if !isempty(elems)
            (L isa Vector{Leaf} && length(L) == 2 && length(elems) == 2 && rshape.kind != "array" &&
             _scalar_matches(L[1], elems[1]) && _scalar_matches(L[2], elems[2])) ||
                return refuse("it returns `$(def.ret_type)`, which the return type's two scalars do not match")
            ret["pass"] = "pair"
        elseif L isa Vector{Leaf} && length(L) == 1 && L[1].offset == 0 && L[1].size == rshape.size &&
               rshape.kind != "array" && _scalar_matches(L[1], def.ret_type)
            ret["pass"] = "direct"
        elseif rshape.kind in _AGGREGATE_KINDS && 0 < rshape.size <= POINTER_BYTES &&
               def.ret_type == "i$(8 * rshape.size)"
            ret["pass"] = "cast"
        else
            return refuse("it returns `$(def.ret_type)`, which does not match its return type")
        end
    end

    # The parameters.
    shapes = Shape[shape(types, p["type"]) for p in dwarf["parameters"]]
    for (k, sh) in enumerate(shapes)
        if dwarf["parameters"][k]["type"] === nothing
            return refuse("parameter `$(params[k]["name"])` has no type in the debug info")
        end
        sh.size < 0 && return refuse("the layout of parameter `$(params[k]["name"])` " *
                                     "(`$(dwarf["parameters"][k]["type"])`) is not known")
    end
    found = _solve(shapes, def.params, first)
    if isempty(found)
        # One pointer more than the debug info lists, at the end: the caller's
        # location, which `#[track_caller]` passes and DWARF does not describe.
        tracked = !isempty(def.params) && def.params[end].type == "ptr" &&
                  length(_solve(shapes, def.params[1:end-1], first)) == 1
        return refuse("its $(length(def.params) - first + 1) IR parameters do not match its " *
                      "$(length(shapes)) parameters in the debug info" *
                      (tracked ? " (one pointer more, at the end: a `#[track_caller]` function)" : ""))
    elseif length(found) > 1
        return refuse("its IR parameters match its parameters in more than one way")
    end
    i = first
    for (k, (pass, n)) in enumerate(found[1])
        params[k]["pass"] = pass
        params[k]["ir"] = collect(i:i+n-1)
        i += n
    end
    return row
end

end # module RustABI
