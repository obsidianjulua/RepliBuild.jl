# GeneratorRust.jl — `RepliBuild.wrap` for a `[rust]` project.
#
# The build wrote, for every function of the crate, how it is called
# (`RustABI.join_function`) and what its types are (`RustDWARF`). This turns
# that into a Julia module: one function per Rust function, one type per Rust
# record or enum.
#
# How a value crosses, in one sentence: it is turned into the bytes Rust lays
# it out as, and the call's arguments are read from those bytes where the
# build's join says each one is.
#
# - A Rust record, tuple or enum with data is a Julia struct of its bytes
#   (`_rb_bytes`), with a property per field read at the offset DWARF gives.
#   The compiler reorders fields and fills niches; bytes at DWARF offsets are
#   right whatever it did, where a Julia struct of fields would have to
#   reproduce its choices.
# - A scalar is itself. `&str` takes a string and `&[T]` a vector, as a
#   pointer and a length. A reference takes a `Ref`, a value (for `&T`) or a
#   pointer.
# - The call is a plain `ccall` where every type of the function's `define`
#   line is a scalar or a pointer and the compiler proved it cannot panic,
#   and the function's thunk otherwise (`RustRuntime.invoke`), inside the
#   library's catching frame when it may panic.
#
# A value that owns something (a `String`, a `Vec`, a `Box`, a record that
# holds one, or any type its author gave drop code) is ended by Rust's drop
# code for its type, once. Which types those are is the compiler's answer,
# read from the build (`metadata["glue"]`, `Builder/RustGlue.jl`), never this
# file's guess:
#
# - a `String` returned is a Julia `String`, and a `Vec` of plain data a Julia
#   `Vector`: copied, and Rust's value dropped at once;
# - any other owning value is a Julia object that holds the Rust value's
#   bytes and drops it when Julia collects the object
#   (`RepliBuild.RustRuntime.drop!` does it sooner). Passing it to a Rust
#   function by value MOVES it: the object is empty afterwards, and using it
#   again is an error, as it is in Rust;
# - `Result<T, E>` returned is the `T`, or a thrown `RustError` holding the `E`.
#
# What is not wrapped is a function that moves a value the build has no drop
# function for (a type no path outside its crate names), and a trait object.
# Such a function is a stub that says why, listed in the module's
# `UNSUPPORTED`. Nothing is guessed, and nothing is silently left out.
#
# Design: docs/updates/RUST_FRONT_END.md.

module GeneratorRust

using Dates

import ..ConfigurationManager
import ..ConfigurationManager: RepliBuildConfig, get_output_path, get_library_name, get_module_name
import ..RustABI
import ..RustGlue
import .._read_json

export generate, wrap_project

# =============================================================================
# Names
# =============================================================================

const _KEYWORDS = Set(["baremodule", "begin", "break", "catch", "const", "continue", "do", "else", "elseif",
                       "end", "export", "false", "finally", "for", "function", "global", "if", "import",
                       "let", "local", "macro", "module", "quote", "return", "struct", "true", "try",
                       "using", "while", "abstract", "mutable", "primitive", "type", "where", "in", "isa",
                       "public", "eval", "include"])

"A Julia identifier from Rust text: paths inside it cut to their last segment, everything else `_`."
function sanitize(text::AbstractString)::String
    s = replace(String(text), r"(?:[A-Za-z_][A-Za-z0-9_]*::)+" => "")
    s = replace(s, r"[^A-Za-z0-9_]+" => "_")
    s = String(strip(s, '_'))
    isempty(s) && (s = "unnamed")
    isdigit(first(s)) && (s = "_" * s)
    return s
end

_reserved(name::AbstractString) = name in _KEYWORDS

# The last segment of a Rust path, generic arguments kept: `a::b::Pair<x::Y>` is `Pair<x::Y>`.
function _last_segment(path::AbstractString)::String
    depth = 0
    last = 1
    i = firstindex(path)
    while i <= lastindex(path)
        c = path[i]
        if c in ('<', '(', '[')
            depth += 1
        elseif c in ('>', ')', ']')
            depth -= 1
        elseif c == ':' && depth == 0 && i < lastindex(path) && path[nextind(path, i)] == ':'
            last = nextind(path, nextind(path, i))
            i = nextind(path, i)
        end
        i = nextind(path, i)
    end
    return String(path[last:end])
end

"""
    function_name(row) -> String

The Julia name of a Rust function, before collisions are settled: a free
function is its own name (`sin`), a method `Type_method` (`Acc_push`), a
trait method `Type_Trait_method` (`Shape_Area_area`), and an instantiation of
a generic carries its arguments (`largest_f64`).
"""
function function_name(row::AbstractDict)::String
    path = String(row["path"])
    m = match(r"^<(.+)>::([^:<>]+)(?:::<(.*)>)?$", path)
    if m !== nothing
        inner = m.captures[1]
        depth = 0
        split_at = 0
        for i in eachindex(inner)
            c = inner[i]
            c in ('<', '(', '[') && (depth += 1)
            c in ('>', ')', ']') && (depth -= 1)
            (depth == 0 && startswith(SubString(inner, i), " as ")) && (split_at = i; break)
        end
        parts = split_at == 0 ? [inner] : [inner[1:prevind(inner, split_at)], inner[split_at+4:end]]
        name = join(vcat([sanitize(_last_segment(p)) for p in parts], [sanitize(m.captures[2])]), "_")
        m.captures[3] === nothing || (name *= "_" * sanitize(m.captures[3]))
        return name
    end
    g = match(r"^(.*)::<(.*)>$", path)
    g === nothing || return sanitize(_last_segment(g.captures[1])) * "_" * sanitize(g.captures[2])
    return sanitize(_last_segment(path))
end

# =============================================================================
# Types
# =============================================================================

"What the generator knows about one Rust type."
struct T
    kind::Symbol                 # :int :float :bool :char :unit :cenum :blob :tuple :array :str :slice :dyn :ptr :option
                                 # :owned :string :vec (a value Rust must drop), :unknown
    jl::String                   # the Julia type a value of it has (for :ptr, the pointee's, or "")
    size::Int
    key::String
    extra::Any
end

mutable struct Ctx
    types::Dict{String,Any}
    names::Dict{String,String}   # type key => generated Julia type name
    memo::Dict{String,T}
    glue::Dict{String,Any}       # type key => the build's drop function for it, and the compiler's "owns"
    refused::Dict{String,Any}    # type key => why the build has none
    helpers::Set{String}         # the root crate's fixed functions the library exports
    boxes::Dict{String,String}   # type key => the Julia name of the `Box` of it, where one is wrapped
end

function Ctx(metadata::AbstractDict)
    types = metadata["types"]
    glue = Dict{String,Any}(String(k) => v for (k, v) in get(metadata, "glue", Dict{String,Any}()))
    ctx = Ctx(types, _type_names(types, glue), Dict{String,T}(), glue,
              Dict{String,Any}(String(k) => v for (k, v) in get(metadata, "glue_refused", Dict{String,Any}())),
              Set{String}(String(h) for h in get(metadata, "glue_helpers", Any[])), Dict{String,String}())
    for key in sort!(collect(keys(glue)))
        t = get(types, key, nothing)
        (t !== nothing && get(t, "kind", "") == "pointer" && get(t, "pointer", "") == "box" &&
         haskey(ctx.names, key) && _drops(ctx, key) && get(t, "target", nothing) isa AbstractString) || continue
        get!(ctx.boxes, String(t["target"]), ctx.names[key])
    end
    return ctx
end

# Whether the compiler said a value of this type must be dropped, and the build has the function that does it.
_drops(ctx::Ctx, key::AbstractString) = (g = get(ctx.glue, key, nothing); g !== nothing && get(g, "owns", false) === true)

_borrowed(key::AbstractString) = startswith(key, "&") || startswith(key, "*")

# The Julia function that wraps one of the root crate's per-type functions: its symbol less one underscore.
_helper(symbol::AbstractString) = String(symbol)[2:end]

const _INT = Dict((true, 1) => "Int8", (true, 2) => "Int16", (true, 4) => "Int32", (true, 8) => "Int64",
                  (true, 16) => "Int128", (false, 1) => "UInt8", (false, 2) => "UInt16", (false, 4) => "UInt32",
                  (false, 8) => "UInt64", (false, 16) => "UInt128")
const _UINT = Dict(1 => "UInt8", 2 => "UInt16", 4 => "UInt32", 8 => "UInt64", 16 => "UInt128")

"The Julia type of an LLVM scalar on a `define` line: what a thunk slot or a `ccall` holds."
function ir_julia(t::AbstractString)::Union{String,Nothing}
    t == "ptr" && return "Base.Ptr{Base.Cvoid}"
    t == "double" && return "Base.Float64"
    t == "float" && return "Base.Float32"
    t == "i1" && return "Base.UInt8"
    m = match(r"^i(\d+)$", t)
    m === nothing && return nothing
    bits = parse(Int, m.captures[1])
    bits % 8 == 0 || return nothing
    haskey(_UINT, bits ÷ 8) && return "Base." * _UINT[bits÷8]
    return "Base.NTuple{$(bits ÷ 8),Base.UInt8}"
end

# The integer a fieldless enum is stored as: the type its DWARF names as its
# own (`core::cmp::Ordering` is an `i8`, and `Less` is -1), unsigned of its
# size when the metadata does not say.
function _enum_base(types::AbstractDict, t::AbstractDict)::String
    u = get(types, something(get(t, "underlying", nothing), ""), nothing)
    signed = u isa AbstractDict && startswith(String(get(u, "encoding", "")), "signed")
    return "Base." * _INT[(signed, Int(t["size"]))]
end

# An enumerator's value as that integer holds it: the same bits, whichever sign the metadata printed.
function _enum_value(value::Integer, base::String, size::Int)
    v = mod(BigInt(value), BigInt(1) << (8 * size))
    (startswith(base, "Base.Int") && v >= BigInt(1) << (8 * size - 1)) && (v -= BigInt(1) << (8 * size))
    return v
end

"The types of a `define` line that a plain `ccall` passes exactly as LLVM does."
_ccall_scalar(t::AbstractString) = t in ("ptr", "double", "float", "i8", "i16", "i32", "i64")

function classify(ctx::Ctx, key::Union{AbstractString,Nothing})::T
    key === nothing && return T(:unit, "Base.Nothing", 0, "", nothing)
    haskey(ctx.memo, key) && return ctx.memo[key]
    ctx.memo[key] = T(:unknown, "", -1, String(key), "a type that refers to itself by value")
    return ctx.memo[key] = _classify(ctx, String(key))
end

function _classify(ctx::Ctx, key::String)::T
    t = get(ctx.types, key, nothing)
    t === nothing && return T(:unknown, "", -1, key, "the type `$key` is not described")
    get(t, "conflict", false) === true && return T(:unknown, "", -1, key, "two different types are named `$key`")
    kind = String(get(t, "kind", "unknown"))
    size = get(t, "size", nothing)
    if kind == "base"
        size isa Integer || return T(:unknown, "", -1, key, "the type `$key` has no size")
        size == 0 && return T(:unit, "Base.Nothing", 0, key, nothing)
        enc = String(get(t, "encoding", "other"))
        enc == "boolean" && size == 1 && return T(:bool, "Base.Bool", 1, key, nothing)
        enc == "utf" && size == 4 && return T(:char, "Base.Char", 4, key, nothing)
        enc == "float" && size in (4, 8) && return T(:float, size == 4 ? "Base.Float32" : "Base.Float64", size, key, nothing)
        if enc in ("signed", "unsigned", "signed_char", "unsigned_char") && haskey(_UINT, size)
            return T(:int, "Base." * _INT[(startswith(enc, "signed"), Int(size))], size, key, nothing)
        end
        return T(:unknown, "", -1, key, "the scalar type `$key` is not one this wraps")
    elseif kind == "pointer"
        target = get(t, "target", nothing)
        pk = String(get(t, "pointer", "other"))
        if pk == "box" && _drops(ctx, key) && haskey(ctx.names, key)
            return T(:owned, ctx.names[key], something(size, RustABI.POINTER_BYTES), key,
                     (drop = String(ctx.glue[key]["drop"]), box = target))
        end
        jl = ""
        if target isa AbstractString
            tt = classify(ctx, target)
            tt.kind in (:int, :float, :bool, :cenum, :blob) && (jl = tt.jl)
        end
        return T(:ptr, jl, something(size, RustABI.POINTER_BYTES), key, pk)
    elseif kind == "array"
        # `[T; N]`: N values of T, one after another.
        elem = get(t, "element", nothing)
        count = get(t, "count", nothing)
        (elem isa AbstractString && count isa Integer && size isa Integer) ||
            return T(:unknown, "", -1, key, "the array `$key` has no length or no size")
        et = classify(ctx, elem)
        et.kind === :unknown && return T(:unknown, "", -1, key, String(et.extra))
        (count == 0 || et.size * count == size) ||
            return T(:unknown, "", -1, key, "the array `$key` is not its elements one after another")
        return T(:array, "Base.NTuple", size, key, (elem = String(elem), count = Int(count), stride = et.size))
    elseif kind == "c_enum"
        (size isa Integer && haskey(_UINT, size)) || return T(:unknown, "", -1, key, "the enum `$key` has no size")
        return T(:cenum, ctx.names[key], size, key, _enum_base(ctx.types, t))
    elseif kind in ("struct", "enum", "union")
        size isa Integer || return T(:unknown, "", -1, key, "the type `$key` has no size")
        fat = String(get(t, "fat", ""))
        fields = get(t, "fields", Any[])
        owned = _drops(ctx, key) && haskey(ctx.names, key)
        if !isempty(fat) && !_borrowed(key)
            # A fat pointer that owns what it points at: `Box<[T]>`, `Box<dyn Trait>`.
            owned && return T(:owned, ctx.names[key], size, key, (drop = String(ctx.glue[key]["drop"]), box = nothing))
            return fat == "dyn" ? T(:dyn, "", size, key, nothing) :
                   T(:unknown, "", -1, key, "a `$key` owns what it points at, and " * _no_drop(ctx, key))
        end
        if fat == "slice"
            ptr_off = fields[1]["offset"]
            len_off = fields[2]["offset"]
            if startswith(key, "&str") || startswith(key, "&mut str")
                return T(:str, "Base.String", size, key, (ptr_off, len_off))
            end
            elem = get(get(ctx.types, fields[1]["type"], Dict()), "target", nothing)
            et = elem isa AbstractString ? classify(ctx, elem) : T(:unknown, "", -1, "", "")
            (startswith(key, "&[") || startswith(key, "&mut [")) && et.kind in (:int, :float) &&
                return T(:slice, et.jl, size, key, (ptr_off, len_off, startswith(key, "&mut")))
            # A slice of plain records is a Julia vector of them: each is its
            # bytes, one after another. Julia aligns a vector's data to 8 at least.
            (startswith(key, "&[") || startswith(key, "&mut [")) && et.kind === :blob && elem isa AbstractString &&
                unmovable(ctx, elem) === nothing && something(get(ctx.types[elem], "align", nothing), 1) <= 8 &&
                return T(:slice, et.jl, size, key, (ptr_off, len_off, startswith(key, "&mut")))
            return T(:unknown, "", -1, key, "a slice of `$(something(elem, "?"))` is not one this wraps yet")
        elseif fat == "dyn"
            return T(:dyn, "", size, key, nothing)
        end
        if get(t, "tuple", false) === true
            elems = Tuple{Int,String}[]
            for f in fields
                (f["offset"] isa Integer && f["type"] isa AbstractString) ||
                    return T(:unknown, "", -1, key, "the tuple `$key` has a field this cannot place")
                push!(elems, (Int(f["offset"]), String(f["type"])))
            end
            return T(:tuple, "Base.Tuple", size, key, elems)
        end
        opt = kind == "enum" ? _option(ctx, t) : nothing
        opt === nothing || return T(:option, ctx.names[key], size, key, opt)
        if owned
            g = ctx.glue[key]
            drop = String(g["drop"])
            if key == "alloc::string::String" && all(h -> h in ctx.helpers, RustGlue.STRING_HELPERS)
                return T(:string, "Base.String", size, key, (drop = drop, name = ctx.names[key]))
            end
            elem = RustGlue.vec_element(ctx.types, key)
            if elem !== nothing && haskey(g, "vec")
                et = classify(ctx, elem)
                if !(et.kind in (:unknown, :dyn, :unit)) && et.size > 0
                    return T(:vec, "Base.Vector", size, key,
                             (drop = drop, view = String(g["vec"]["view"]), new = String(g["vec"]["new"]),
                              elem = elem, stride = et.size, name = ctx.names[key]))
                end
            end
            return T(:owned, ctx.names[key], size, key, (drop = drop, box = nothing))
        end
        return T(:blob, ctx.names[key], size, key, kind)
    end
    return T(:unknown, "", -1, key, "the type `$key` is not one this wraps")
end

# A discriminant's value as a literal of the unsigned integer of `size` bytes.
# The metadata carries one that may not fit `Int64` as a decimal string.
function _discr_literal(value, size)::Union{String,Nothing}
    haskey(_UINT, size) || return nothing
    v = value isa Integer ? BigInt(value) : value isa AbstractString ? tryparse(BigInt, value) : nothing
    v === nothing && return nothing
    return "Base.$(_UINT[size])(0x$(string(mod(v, BigInt(1) << (8 * size)); base = 16)))"
end

# Where an enum's discriminant is and how wide: (offset, bytes), or `nothing`.
function _discriminant(ctx::Ctx, t::AbstractDict)
    d = get(t, "discriminant", nothing)
    (d isa AbstractDict && get(d, "offset", nothing) isa Integer) || return nothing
    size = get(d, "size", nothing)
    size isa Integer || (size = get(get(ctx.types, something(get(d, "type", nothing), ""), Dict()), "size", nothing))
    (size isa Integer && haskey(_UINT, size)) || return nothing
    return (offset = Int(d["offset"]), size = Int(size))
end

# `Option<P>`: the payload's key and offset, and how `None` is told from `Some`:
# - `:tag`: a discriminant of its own, with a value for each;
# - `:null`: `None` is the null pointer of a payload that is a pointer;
# - `:niche`: `None` is one value, at the discriminant's place, that the
#   payload's own bytes never have (`Option<String>`).
function _option(ctx::Ctx, t::AbstractDict)
    RustGlue.payload_enum(t) === :option || return nothing
    variants = get(t, "variants", Any[])
    length(variants) == 2 || return nothing
    none = findfirst(v -> get(v, "name", "") == "None", variants)
    some = findfirst(v -> get(v, "name", "") == "Some", variants)
    (none === nothing || some === nothing) && return nothing
    d = get(t, "discriminant", nothing)
    d === nothing && return nothing
    sv = get(ctx.types, get(variants[some], "type", ""), nothing)
    sv === nothing && return nothing
    f = get(sv, "fields", Any[])
    (length(f) == 1 && f[1]["offset"] isa Integer && f[1]["type"] isa AbstractString) || return nothing
    payload = String(f[1]["type"])
    nv = get(variants[none], "discr_value", nothing)
    if get(variants[some], "default", false) === true
        pk = get(get(ctx.types, payload, Dict()), "kind", "")
        (pk == "pointer" && nv == 0 && f[1]["offset"] == d["offset"]) &&
            return (payload = payload, offset = Int(f[1]["offset"]), mode = :null)
        place = _discriminant(ctx, t)
        lit = place === nothing ? nothing : _discr_literal(nv, place.size)
        lit === nothing && return nothing
        return (payload = payload, offset = Int(f[1]["offset"]), mode = :niche,
                tag_jl = "Base." * _UINT[place.size], tag_offset = place.offset, none = lit)
    end
    sv_value = get(variants[some], "discr_value", nothing)
    (nv isa Integer && sv_value isa Integer && d["type"] isa AbstractString && d["offset"] isa Integer) || return nothing
    return (payload = payload, offset = Int(f[1]["offset"]), mode = :tag, tag = String(d["type"]),
            tag_offset = Int(d["offset"]), none = Int(nv), some = Int(sv_value))
end

# Whether an `Option` is the nullable reference a caller passes as `nothing`, a `Ref` or a pointer.
_nullable_reference(ctx::Ctx, t::T) =
    t.kind === :option && t.extra.mode === :null && classify(ctx, t.extra.payload).kind === :ptr

# `Result<V, E>`: each side's payload (its key and offset; no key for none),
# and the test that tells `Ok`: the discriminant equal to `Ok`'s value, or
# different from `Err`'s when `Ok` is the variant with no value of its own.
function _result(ctx::Ctx, t::AbstractDict)
    RustGlue.payload_enum(t) === :result || return nothing
    variants = get(t, "variants", Any[])
    length(variants) == 2 || return nothing
    ok = findfirst(v -> get(v, "name", "") == "Ok", variants)
    err = findfirst(v -> get(v, "name", "") == "Err", variants)
    (ok === nothing || err === nothing) && return nothing
    place = _discriminant(ctx, t)
    place === nothing && return nothing
    function side(v)
        vt = get(ctx.types, get(v, "type", ""), nothing)
        vt === nothing && return nothing
        f = get(vt, "fields", Any[])
        isempty(f) && return (key = nothing, offset = 0)
        (length(f) == 1 && f[1]["offset"] isa Integer && f[1]["type"] isa AbstractString) || return nothing
        return (key = String(f[1]["type"]), offset = Int(f[1]["offset"]))
    end
    (okp, errp) = (side(variants[ok]), side(variants[err]))
    (okp === nothing || errp === nothing) && return nothing
    tag = "Base." * _UINT[place.size]
    if haskey(variants[ok], "discr_value")
        lit = _discr_literal(variants[ok]["discr_value"], place.size)
        lit === nothing && return nothing
        return (ok = okp, err = errp, tag_jl = tag, tag_offset = place.offset, test = "== " * lit)
    elseif haskey(variants[err], "discr_value")
        lit = _discr_literal(variants[err]["discr_value"], place.size)
        lit === nothing && return nothing
        return (ok = okp, err = errp, tag_jl = tag, tag_offset = place.offset, test = "!= " * lit)
    end
    return nothing
end

"""
    owns(ctx, key) -> Bool

Whether a value of this type, held by value, MAY own memory, judged from its
shape alone: a `Box`, or any pointer inside a record that is not a plain
reference or a fat pointer, or anything of `alloc`. Conservative: a record
with a raw pointer field counts. This is the fallback for a type the
compiler was not asked about (`unmovable`): the compiler's own answer, where
the build has it, is the one used.
"""
function owns(ctx::Ctx, key::Union{AbstractString,Nothing}, depth::Int = 0, top::Bool = true)::Bool
    key === nothing && return false
    depth > 16 && return true
    t = get(ctx.types, key, nothing)
    t === nothing && return true
    kind = get(t, "kind", "unknown")
    kind in ("base", "c_enum") && return false
    if kind == "pointer"
        pk = get(t, "pointer", "other")
        pk == "box" && return true
        pk in ("ref", "mut_ref") && return false
        return !top                       # a raw pointer is the caller's business alone; inside a record it may be owned
    end
    startswith(key, "alloc::") && return true
    isempty(get(t, "fat", "")) || return !(startswith(key, "&") || startswith(key, "*"))
    if kind == "array"
        return owns(ctx, get(t, "element", nothing), depth + 1, false)
    elseif kind in ("struct", "enum", "union")
        for f in get(t, "fields", Any[])
            owns(ctx, get(f, "type", nothing), depth + 1, false) && return true
        end
        for v in get(t, "variants", Any[])
            haskey(v, "type") && owns(ctx, v["type"], depth + 1, false) && return true
        end
        return false
    end
    return true
end

_no_drop(ctx::Ctx, key::AbstractString) =
    haskey(ctx.refused, key) ? "the build has no drop function for it: $(ctx.refused[key])" :
                               "the library was built without a drop function for it (build it again)"

"""
    unmovable(ctx, key) -> Union{String,Nothing}

Why a value of this type cannot be moved between Julia and Rust, or
`nothing` when it can. A type the compiler answered for is movable: it is
plain data, or the build has the function that drops it. A type it was not
asked about, or could not be (no path outside the crate names it), is
movable only when its shape shows nothing it could own (`owns`).
"""
function unmovable(ctx::Ctx, key::Union{AbstractString,Nothing}, depth::Int = 0)::Union{String,Nothing}
    key === nothing && return nothing
    depth > 16 && return "a `$key`, which is nested too deeply to read"
    t = get(ctx.types, key, nothing)
    t === nothing && return "a `$key`, which is not described"
    kind = get(t, "kind", "unknown")
    kind in ("base", "c_enum") && return nothing
    haskey(ctx.glue, key) && return nothing
    if kind == "pointer"
        return get(t, "pointer", "other") == "box" ? "a `$key`, which owns its value, and " * _no_drop(ctx, key) : nothing
    end
    isempty(get(t, "fat", "")) ||
        return _borrowed(key) ? nothing : "a `$key`, which owns what it points at, and " * _no_drop(ctx, key)
    if kind == "array"
        return unmovable(ctx, get(t, "element", nothing), depth + 1)
    elseif kind in ("struct", "enum", "union")
        if get(t, "tuple", false) === true
            for f in get(t, "fields", Any[])
                why = unmovable(ctx, get(f, "type", nothing), depth + 1)
                why === nothing || return why
            end
            return nothing
        end
        return owns(ctx, key) ? "a `$key`, which may own memory, and " * _no_drop(ctx, key) : nothing
    end
    return "a `$key`, which is not a type this moves"
end

"The Julia type a value of this type is read as: an element type, or a docstring's."
function value_type(ctx::Ctx, key::Union{AbstractString,Nothing})::String
    t = classify(ctx, key)
    k = t.kind
    k in (:int, :float, :bool, :char, :unit, :cenum, :blob, :owned, :string, :str) && return t.jl
    if k === :vec
        return decode(ctx, t.extra.elem, "_e", 0; take = false) === nothing ? t.extra.name :
               "Base.Vector{$(value_type(ctx, t.extra.elem))}"
    end
    k === :option && return "Base.Union{Base.Nothing,$(value_type(ctx, t.extra.payload))}"
    k === :tuple && return "Base.Tuple{" * join((value_type(ctx, ek) for (_, ek) in t.extra), ",") * "}"
    k === :array && return "Base.NTuple{$(t.extra.count),$(value_type(ctx, t.extra.elem))}"
    return "Base.Any"
end

# A buffer Julia allocates is aligned to 16 bytes. A type Rust aligns more
# strictly cannot be built in one or returned into one.
_overaligned(ctx::Ctx, key) = key isa AbstractString && something(get(get(ctx.types, key, Dict()), "align", nothing), 1) > 16

# The Julia names of the types the module defines: records, enums, unions and
# fieldless enums, and a `Box` or an owning fat pointer the build has a drop
# function for. A variant's record is named after its enum (`Shape_Circle`).
# The default allocator is not part of a name (`Vec_f64`), and Rust's
# `String` is `RustString`: the module's own code says `Base.String`, and a
# reader should not have to.
function _type_names(types::AbstractDict, glue::AbstractDict = Dict{String,Any}())::Dict{String,String}
    names = Dict{String,String}()
    taken = Set{String}()
    for key in sort!(collect(keys(types)))
        t = types[key]
        kind = get(t, "kind", "")
        scope = String[]
        if kind == "pointer"
            (get(t, "pointer", "") == "box" && haskey(glue, key)) || continue
            base = sanitize(RustGlue.source_name(key))
        else
            kind in ("struct", "enum", "union", "c_enum") || continue
            get(t, "tuple", false) === true && continue
            isempty(get(t, "fat", "")) || (haskey(glue, key) && !_borrowed(key)) || continue
            scope = String[String(x) for x in get(t, "scope", Any[])]
            parent = join(scope, "::")
            base = key == "alloc::string::String" ? "RustString" :
                   sanitize(RustGlue.source_name(String(get(t, "name", key))))
            haskey(types, parent) && get(types[parent], "kind", "") == "enum" &&
                (base = sanitize(RustGlue.source_name(String(get(types[parent], "name", parent)))) * "_" * base)
        end
        name = base
        # Two types with one last segment: the scope tells them apart.
        k = length(scope)
        while (name in taken || _reserved(name)) && k >= 1
            name = sanitize(scope[k]) * "_" * name
            k -= 1
        end
        n = 2
        while name in taken || _reserved(name)
            name = base * "_" * string(n)
            n += 1
        end
        push!(taken, name)
        names[String(key)] = name
    end
    return names
end

# =============================================================================
# Values in memory: decode and encode
# =============================================================================

_at(p::String, off::Int) = off == 0 ? p : "$p + $off"
_load(jl::String, p::String, off::Int) = "Base.unsafe_load(Base.Ptr{$jl}($(_at(p, off))))"
_store(jl::String, p::String, off::Int, v::String) = "Base.unsafe_store!(Base.Ptr{$jl}($(_at(p, off))), $v)"

# A handle that takes over the Rust value whose bytes are at `p + off`.
_adopt(name::String, size::Int, p::String, off::Int) =
    "$name(_RbRaw(), " * _load("Base.NTuple{$size,Base.UInt8}", p, off) * ")"

"""
    decode(ctx, key, p, off; take) -> Union{String,Nothing}

The expression that reads a value of type `key` from memory at `p + off`, or
`nothing` when this cannot.

`take` says whose the value is afterwards. With `take`, Julia takes it over
(a returned value): a `String` or a `Vec` of readable elements is copied and
dropped there and then, and any other owning value becomes the Julia object
that holds it. Without, the value stays where it is (a field, an element):
what can be copied is copied, and an owning value that cannot be is not
read at all, since two holders of one value would each drop it.
"""
function decode(ctx::Ctx, key::Union{AbstractString,Nothing}, p::String, off::Int; take::Bool)::Union{String,Nothing}
    t = classify(ctx, key)
    k = t.kind
    k === :unit && return "nothing"
    k in (:int, :float) && return _load(t.jl, p, off)
    k === :bool && return "(" * _load("Base.UInt8", p, off) * " != 0x00)"
    k === :char && return "Base.Char(" * _load("Base.UInt32", p, off) * ")"
    k === :cenum && return "$(t.jl)(" * _load(t.extra, p, off) * ")"
    if k === :blob
        # A copy of bytes that hold an owning pointer would outlive what it points at.
        unmovable(ctx, key) === nothing || return nothing
        return _load(t.jl, p, off)
    elseif k === :ptr
        t.extra == "box" && return nothing
        return "Base.Ptr{$(isempty(t.jl) ? "Base.Cvoid" : t.jl)}(" * _load("Base.UInt", p, off) * ")"
    elseif k === :str
        (po, lo) = t.extra
        return "Base.unsafe_string(Base.Ptr{Base.UInt8}(" * _load("Base.UInt", p, off + po) * "), " *
               _load("Base.UInt", p, off + lo) * ")"
    elseif k === :slice
        (po, lo, _) = t.extra
        return "Base.copy(Base.unsafe_wrap(Base.Array, Base.Ptr{$(t.jl)}(" * _load("Base.UInt", p, off + po) * "), " *
               _load("Base.UInt", p, off + lo) * "))"
    elseif k === :tuple
        parts = String[]
        for (eo, ek) in t.extra
            d = decode(ctx, ek, p, off + eo; take)
            d === nothing && return nothing
            push!(parts, d)
        end
        return "(" * join(parts, ", ") * (length(parts) == 1 ? "," : "") * ")"
    elseif k === :array
        # The place is bound before the elements are read from it: an array of arrays reads through its own.
        a = t.extra
        inner = decode(ctx, a.elem, "(_a + (_i - 1) * $(a.stride))", 0; take)
        inner === nothing && return nothing
        return "(let _a = $(_at(p, off)); Base.ntuple(_i -> $inner, Base.Val($(a.count))) end)"
    elseif k === :option
        # `None` holds nothing, and a `Some` is its payload: taking the payload takes all of it.
        o = t.extra
        inner = decode(ctx, o.payload, p, off + o.offset; take)
        inner === nothing && return nothing
        o.mode === :null && return "(" * _load("Base.UInt", p, off + o.offset) * " == 0 ? nothing : $inner)"
        o.mode === :niche && return "(" * _load(o.tag_jl, p, off + o.tag_offset) * " == $(o.none) ? nothing : $inner)"
        tag = classify(ctx, o.tag)
        tag.kind === :int || return nothing
        return "(" * _load(tag.jl, p, off + o.tag_offset) * " == $(o.some) ? $inner : nothing)"
    elseif k === :owned
        return take ? _adopt(t.jl, t.size, p, off) : nothing
    elseif k === :string
        return (take ? "_rb_string_take!(" : "_rb_string_at(") * _at(p, off) * ")"
    elseif k === :vec
        e = t.extra
        inner = decode(ctx, e.elem, "_e", 0; take = false)
        inner === nothing && return take ? _adopt(e.name, t.size, p, off) : nothing
        et = classify(ctx, e.elem)
        view = _helper(e.view)
        read(q) = et.kind in (:int, :float, :blob) ?
            "RepliBuild.RustRuntime.vec_copy($(et.jl), $view($q)...)" :
            "RepliBuild.RustRuntime.vec_read($(value_type(ctx, e.elem)), _e -> $inner, $view($q)..., $(e.stride))"
        take || return read(_at(p, off))
        return "(let _q = $(_at(p, off)); _v = $(read("_q")); $(_helper(e.drop))(_q); _v end)"
    end
    return nothing
end

# Whether a `Vec` of this type can be built from a Julia vector: its elements can be written.
_vec_buildable(ctx::Ctx, t::T) = encode(ctx, t.extra.elem, "_e", 0, "_x") !== nothing

"""
    encode(ctx, key, p, off, v) -> Union{Vector{String},Nothing}

The statements that write `v` as a value of type `key` to memory at
`p + off`, or `nothing`. What is written belongs to whoever reads that
memory next: an owning Julia object written here is moved (it is empty
afterwards), and a `String` or `Vec` written from a Julia value is built by
Rust for that reader to drop.
"""
function encode(ctx::Ctx, key::Union{AbstractString,Nothing}, p::String, off::Int, v::String)::Union{Vector{String},Nothing}
    t = classify(ctx, key)
    k = t.kind
    k === :unit && return String[]
    k in (:int, :float) && return [_store(t.jl, p, off, "Base.convert($(t.jl), $v)")]
    k === :bool && return [_store("Base.UInt8", p, off, "Base.UInt8($v::Base.Bool)")]
    k === :char && return [_store("Base.UInt32", p, off, "Base.UInt32(Base.Char($v))")]
    k === :cenum && return [_store(t.extra, p, off, "$(t.extra)(Base.Integer($v::$(t.jl)))")]
    if k === :blob
        unmovable(ctx, key) === nothing || return nothing
        return [_store(t.jl, p, off, "$v::$(t.jl)")]
    end
    bytes = "Base.NTuple{$(t.size),Base.UInt8}"
    k === :owned && return [_store(bytes, p, off, "_rb_take!($v::$(t.jl))")]
    k === :string && return ["_rb_string_new!($(_at(p, off)), $v)"]
    if k === :vec
        return _vec_buildable(ctx, t) ? ["$(_helper(t.extra.new))!($(_at(p, off)), $v)"] :
               [_store(bytes, p, off, "_rb_take!($v::$(t.extra.name))")]
    elseif k === :array
        a = t.extra
        inner = encode(ctx, a.elem, "(_a + (_i - 1) * $(a.stride))", 0, "_t[_i]")
        inner === nothing && return nothing
        return ["let _a = $(_at(p, off)), _t = $v",
                "    Base.length(_t) == $(a.count) || Base.throw(Base.ArgumentError(\"a Rust `$(key)` takes $(a.count) values\"))",
                "    for _i in 1:$(a.count)",
                ["        " * s for s in inner]...,
                "    end",
                "end"]
    elseif k === :tuple
        out = String[]
        for (i, (eo, ek)) in enumerate(t.extra)
            e = encode(ctx, ek, p, off + eo, "$v[$i]")
            e === nothing && return nothing
            append!(out, e)
        end
        return out
    elseif k === :option
        o = t.extra
        inner = encode(ctx, o.payload, p, off + o.offset, v)
        inner === nothing && return nothing
        body = ["    " * s for s in inner]
        if o.mode === :tag
            tag = classify(ctx, o.tag)
            tag.kind === :int || return nothing
            return ["if $v === nothing",
                    "    " * _store(tag.jl, p, off + o.tag_offset, "Base.convert($(tag.jl), $(o.none))"),
                    "else",
                    "    " * _store(tag.jl, p, off + o.tag_offset, "Base.convert($(tag.jl), $(o.some))"),
                    body...,
                    "end"]
        elseif o.mode === :null
            _nullable_reference(ctx, t) && return nothing      # a reference is passed, never built
            return ["if $v === nothing", "    " * _store("Base.UInt", p, off + o.offset, "Base.UInt(0)"),
                    "else", body..., "end"]
        elseif o.mode === :niche
            return ["if $v === nothing", "    " * _store(o.tag_jl, p, off + o.tag_offset, o.none),
                    "else", body..., "end"]
        end
    end
    return nothing
end

"The type a Julia argument for `key` is declared with, and how the docstring spells it."
function julia_arg_type(ctx::Ctx, key)::String
    t = classify(ctx, key)
    k = t.kind
    k === :int && return "Base.Integer"
    k === :float && return "Base.Real"
    k === :bool && return "Base.Bool"
    k === :char && return "Base.AbstractChar"
    k in (:cenum, :blob, :owned) && return t.jl
    k === :tuple && return "Base.Tuple"
    k === :array && return "Base.Union{Base.Tuple,Base.AbstractVector}"
    k === :str && return "Base.AbstractString"
    k === :slice && return "Base.DenseVector{$(t.jl)}"
    k === :string && return "Base.Union{Base.AbstractString,$(t.extra.name)}"
    k === :vec && return _vec_buildable(ctx, t) ? "Base.Union{Base.AbstractVector,$(t.extra.name)}" : t.extra.name
    if k === :option
        _nullable_reference(ctx, t) && return "Base.Union{Base.Nothing,Base.RefValue,Base.Ptr}"
        return "Base.Union{Base.Nothing,$(julia_arg_type(ctx, t.extra.payload))}"
    end
    return "Base.Any"
end

# =============================================================================
# One function
# =============================================================================

struct Emitted
    name::String
    text::String
    tier::String              # "ccall", "thunk", "guarded" or "unsupported"
    reason::String
end

_sym(s::AbstractString) = repr(Symbol(s))

"""
    emit_function(ctx, row, name, index) -> Emitted

The Julia function for one row. `tier` says how it calls: a plain `ccall`,
its thunk, its thunk inside the catching frame (`guarded`), or not at all
(`unsupported`, with the reason, and a stub that raises it).
"""
function emit_function(ctx::Ctx, row::AbstractDict, name::String, index::Int)::Emitted
    path = String(row["path"])
    unsupported(reason) = Emitted(name, _stub(name, path, reason), "unsupported", reason)
    row["refusal"] === nothing || return unsupported(String(row["refusal"]))
    # Rust lets nothing call a destructor by name: a value dropped that way would be dropped again.
    occursin(r" as core::ops::drop::Drop>::drop$", path) &&
        return unsupported("Rust does not let a destructor be called directly; " *
                           "`RepliBuild.RustRuntime.drop!` ends the value this is the destructor of")
    ir = row["ir"]
    irp = ir["parameters"]
    ret = row["return"]
    rpass = String(get(ret, "pass", ""))

    prelude = String[]            # statements before the call
    cleanup = String[]            # statements after a call that returned
    keep = String[]               # what GC.@preserve holds
    values = Vector{Union{Nothing,Tuple{String,String}}}(nothing, length(irp))   # per IR parameter: (Julia type, expression)
    args = String[]               # the Julia signature
    docs = String[]

    for (k, p) in enumerate(row["parameters"])
        pass = String(get(p, "pass", ""))
        t = classify(ctx, p["type"])
        pname = "a$k"
        shown = sanitize(String(p["name"]))
        pass == "ignore" && continue                      # a zero-sized value is not passed: it is not an argument here
        t.kind === :unknown && return unsupported(String(t.extra))
        irs = Int[Int(i) for i in p["ir"]]
        irt = String[String(irp[i]["type"]) for i in irs]
        any(x -> ir_julia(x) === nothing, irt) &&
            return unsupported("a parameter is passed as `$(join(irt, ", "))`, which this does not carry")
        push!(docs, "$shown::$(p["type"])")
        if t.kind in (:int, :float, :bool, :char, :cenum) && pass == "direct"
            jt = ir_julia(irt[1])
            expr = t.kind === :int ? "Base.reinterpret($jt, Base.convert($(t.jl), $pname))" :
                   t.kind === :float ? "Base.convert($(t.jl), $pname)" :
                   t.kind === :bool ? "Base.UInt8($pname)" :
                   t.kind === :char ? "Base.UInt32(Base.Char($pname))" :
                                      "Base.reinterpret($jt, $(t.extra)(Base.Integer($pname)))"
            # A scalar wider or narrower than its IR type is not a direct scalar this reads.
            (t.kind in (:int, :cenum) && t.size != something(RustABI._ir_scalar_bytes(irt[1]), -1)) &&
                return unsupported("the parameter `$shown` is passed as `$(irt[1])`, not at its own width")
            values[irs[1]] = (jt, expr)
            push!(args, "$pname::$(julia_arg_type(ctx, p["type"]))")
        elseif t.kind in (:str, :slice) && pass == "pair"
            if t.kind === :str
                push!(prelude, "$(pname)_s = Base.String($pname)")
                push!(prelude, "Base.isvalid($(pname)_s) || Base.throw(Base.ArgumentError(\"a Rust `&str` must be valid UTF-8\"))")
                values[irs[1]] = ("Base.Ptr{Base.Cvoid}", "Base.Ptr{Base.Cvoid}(Base.pointer($(pname)_s))")
                values[irs[2]] = ("Base.UInt64", "Base.UInt64(Base.ncodeunits($(pname)_s))")
                push!(keep, "$(pname)_s")
            else
                values[irs[1]] = ("Base.Ptr{Base.Cvoid}", "Base.Ptr{Base.Cvoid}(Base.pointer($pname))")
                values[irs[2]] = ("Base.UInt64", "Base.UInt64(Base.length($pname))")
                push!(keep, pname)
            end
            push!(args, "$pname::$(julia_arg_type(ctx, p["type"]))")
        elseif t.kind === :ptr && pass == "direct"
            t.extra == "box" &&
                return unsupported("the parameter `$shown` is a `Box`, which owns its value, and " * _no_drop(ctx, String(p["type"])))
            target = get(ctx.types[p["type"]], "target", nothing)
            tt = target isa AbstractString ? classify(ctx, target) : T(:unknown, "", -1, "", "")
            # A `Box` of the target is a pointer to one: it is passed where a reference is taken.
            box = target isa AbstractString ? get(ctx.boxes, String(target), "") : ""
            address = "_rb_address($(pname)_r)"
            if tt.kind in (:owned, :string, :vec)
                # A reference to a value Rust must drop: the Julia object that
                # holds one. A shared reference to a `String` or a `Vec` also
                # takes a Julia value, as an object of its own for this call.
                handle = tt.kind === :owned ? tt.jl : tt.extra.name
                from = t.extra != "ref" ? "" : tt.kind === :string ? "Base.AbstractString" :
                       (tt.kind === :vec && _vec_buildable(ctx, tt)) ? "Base.AbstractVector" : ""
                if isempty(from)
                    push!(prelude, "$(pname)_r = $pname")
                    push!(args, "$pname::Base.Union{$handle,Base.Ptr}")
                else
                    push!(prelude, "$(pname)_r = $pname isa $from ? $handle($pname) : $pname")
                    push!(cleanup, "$pname isa $from && RepliBuild.RustRuntime.drop!($(pname)_r)")
                    push!(args, "$pname::Base.Union{$handle,$from,Base.Ptr}")
                end
            elseif t.extra == "ref" && !isempty(t.jl)
                push!(prelude, "$(pname)_r = ($pname isa Base.Ptr || $pname isa Base.RefValue" *
                               (isempty(box) ? "" : " || $pname isa $box") * ") ? $pname : Base.RefValue{$(t.jl)}($pname)")
                push!(args, "$pname::Base.Union{$(t.jl),Base.RefValue{$(t.jl)},Base.Ptr" * (isempty(box) ? "" : ",$box") * "}")
            else
                push!(prelude, "$(pname)_r = $pname")
                push!(args, "$pname::Base.Union{Base.RefValue,Base.Ptr" * (isempty(box) ? "" : ",$box") * "}")
            end
            (isempty(box) || tt.kind in (:owned, :string, :vec)) ||
                (address = "($(pname)_r isa $box ? _rb_target($(pname)_r) : $address)")
            push!(keep, "$(pname)_r")
            values[irs[1]] = ("Base.Ptr{Base.Cvoid}", address)
        elseif _nullable_reference(ctx, t) && pass == "direct"
            push!(prelude, "$(pname)_r = $pname")
            push!(keep, "$(pname)_r")
            values[irs[1]] = ("Base.Ptr{Base.Cvoid}", "($(pname)_r === nothing ? Base.C_NULL : _rb_address($(pname)_r))")
            push!(args, "$pname::$(julia_arg_type(ctx, p["type"]))")
        elseif t.kind in (:blob, :tuple, :array, :option, :int, :float, :bool, :char, :cenum, :owned, :string, :vec)
            # By value, as its bytes; the IR arguments are read from them. A
            # value that owns something is MOVED: Rust drops it from here on.
            why = unmovable(ctx, p["type"])
            why === nothing || return unsupported("the parameter `$shown` moves $why")
            _overaligned(ctx, p["type"]) && return unsupported("the parameter `$shown` is aligned beyond 16 bytes")
            t.size > 0 || return unsupported("the parameter `$shown` has no size")
            enc = encode(ctx, p["type"], "$(pname)_p", 0, pname)
            enc === nothing && return unsupported("a `$(p["type"])` cannot be built from a Julia value yet")
            push!(prelude, "$(pname)_b = Base.RefValue{Base.NTuple{$(t.size),Base.UInt8}}(Base.ntuple(_ -> 0x00, Base.Val($(t.size))))")
            push!(prelude, "$(pname)_p = Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{Base.NTuple{$(t.size),Base.UInt8}}, $(pname)_b))")
            push!(prelude, "GC.@preserve $(pname)_b begin")
            append!(prelude, ["    " * s for s in enc])
            push!(prelude, "end")
            push!(keep, "$(pname)_b")
            if pass == "indirect"
                values[irs[1]] = ("Base.Ptr{Base.Cvoid}", "$(pname)_p")
            elseif pass in ("direct", "cast")
                values[irs[1]] = (ir_julia(irt[1]), _load(ir_julia(irt[1]), "$(pname)_p", 0))
            elseif pass == "pair"
                leaves = RustABI.shape(ctx.types, p["type"]).leaves
                (leaves isa Vector{RustABI.Leaf} && length(leaves) == 2) ||
                    return unsupported("the two parts of the parameter `$shown` are not placed")
                for (j, leaf) in enumerate(leaves)
                    values[irs[j]] = (ir_julia(irt[j]), _load(ir_julia(irt[j]), "$(pname)_p", leaf.offset))
                end
            else
                return unsupported("the parameter `$shown` is passed as `$pass`, which this does not carry")
            end
            push!(args, "$pname::$(julia_arg_type(ctx, p["type"]))")
        else
            return unsupported(t.kind === :dyn ? "the parameter `$shown` is a trait object, which is not wrapped yet" :
                               "the parameter `$shown` (`$(p["type"])`) is passed in a way this does not carry yet")
        end
    end

    # The return.
    rt = classify(ctx, ret["type"])
    sret = rpass == "indirect"
    ret_ir = String(ir["return"]["type"])
    rsize = 0
    result = "nothing"
    if rpass != "ignore"
        rt.kind === :unknown && return unsupported(String(rt.extra))
        rt.kind === :dyn && return unsupported("it returns a trait object, which is not wrapped yet")
        if !(rt.kind in (:str, :slice)) && !(rt.kind === :ptr && rt.extra != "box")
            why = unmovable(ctx, ret["type"])
            why === nothing || return unsupported("it returns $why")
        end
        _overaligned(ctx, ret["type"]) && return unsupported("it returns a type aligned beyond 16 bytes")
        rsize = rt.size
        # `Result<V, E>`: the `V` of an `Ok`, or the `E` of an `Err`, thrown.
        res = _result(ctx, get(ctx.types, something(ret["type"], ""), Dict{String,Any}()))
        read = nothing
        if res !== nothing
            okx = res.ok.key === nothing ? "nothing" : decode(ctx, res.ok.key, "_ret_p", res.ok.offset; take = true)
            errx = res.err.key === nothing ? "nothing" : decode(ctx, res.err.key, "_ret_p", res.err.offset; take = true)
            (okx === nothing || errx === nothing) ||
                (read = "(" * _load(res.tag_jl, "_ret_p", res.tag_offset) * " $(res.test) ? $okx : " *
                        "Base.throw(RepliBuild.RustRuntime.RustError($errx)))")
        end
        read === nothing && (read = decode(ctx, ret["type"], "_ret_p", 0; take = true))
        read === nothing && return unsupported("a `$(ret["type"])` cannot be read back yet")
        result = read
    end
    start = sret ? 2 : 1
    for i in start:length(irp)
        values[i] === nothing && return unsupported("the IR parameter $i is not accounted for")
    end

    plain = row["nounwind"] === true && !sret && all(i -> _ccall_scalar(irp[i]["type"]), start:length(irp)) &&
            (ret_ir == "void" || _ccall_scalar(ret_ir)) && rpass in ("ignore", "direct")
    guarded = row["nounwind"] !== true
    tier = plain ? "ccall" : guarded ? "guarded" : "thunk"

    io = IOBuffer()
    sig = isempty(docs) ? "" : join(docs, ", ")
    println(io, "\"\"\"")
    println(io, "    $name($sig)", ret["type"] === nothing ? "" : " -> $(ret["type"])")
    println(io)
    println(io, "Rust: `$path`", get(row, "own_debug_info", true) === false ?
                " (merged with another function by the compiler: the types shown are that function's)" : "",
            get(row, "instance", "") == "inline" ? " (an inline function: the library has it because the build named it)" :
            get(row, "instance", "") == "instantiate" ? " (compiled for these types because `[rust] instantiate` asks for it)" : "")
    println(io, "\"\"\"")
    println(io, "function $name(", join(args, ", "), ")")
    for s in prelude
        println(io, "    ", s)
    end
    vals = [values[i] for i in start:length(irp)]
    # The result, then what a returned call leaves to tidy (an object made for this call alone).
    function finish(indent, value)
        if isempty(cleanup)
            println(io, indent, "return $value")
        else
            println(io, indent, "_out = $value")
            foreach(c -> println(io, indent, c), cleanup)
            println(io, indent, "return _out")
        end
    end
    if plain
        types = "(" * join((v[1] for v in vals), ", ") * (length(vals) == 1 ? "," : "") * ")"
        rjl = ret_ir == "void" ? "Base.Cvoid" : ir_julia(ret_ir)
        # A function with no symbol is called where the library's static says it is.
        callee = haskey(row, "address") ?
            "Base.unsafe_load(Base.cglobal(($(_sym(row["address"])), LIBRARY_PATH), Base.Ptr{Base.Cvoid}))" :
            "($(_sym(row["symbol"])), LIBRARY_PATH)"
        call = "ccall($callee, $rjl, $types" *
               (isempty(vals) ? "" : ", " * join((v[2] for v in vals), ", ")) * ")"
        if rpass == "ignore"
            println(io, isempty(keep) ? "    $call" : "    GC.@preserve $(join(keep, " ")) $call")
            finish("    ", "nothing")
        else
            println(io, "    _ret = Base.RefValue{Base.NTuple{$(max(rsize, 1)),Base.UInt8}}()")
            println(io, "    _r = ", isempty(keep) ? call : "GC.@preserve $(join(keep, " ")) $call")
            println(io, "    GC.@preserve _ret begin")
            println(io, "        _ret_p = Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{Base.NTuple{$(max(rsize, 1)),Base.UInt8}}, _ret))")
            println(io, "        ", _store(rjl, "_ret_p", 0, "_r"))
            finish("        ", result)
            println(io, "    end")
        end
    else
        for (j, v) in enumerate(vals)
            println(io, "    _v$j = Base.RefValue{$(v[1])}($(v[2]))")
        end
        println(io, "    _ret = Base.RefValue{Base.NTuple{$(max(rsize, 1)),Base.UInt8}}()")
        held = join(vcat(keep, ["_v$j" for j in eachindex(vals)], ["_ret"]), " ")
        println(io, "    GC.@preserve $held begin")
        println(io, "        _ret_p = Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{Base.NTuple{$(max(rsize, 1)),Base.UInt8}}, _ret))")
        slots = join(("Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{$(v[1])}, _v$j))" for (j, v) in enumerate(vals)), ", ")
        println(io, "        _slots = Base.Ptr{Base.Cvoid}[$slots]")
        println(io, "        RepliBuild.RustRuntime.invoke(_rb_library(), RepliBuild.RustRuntime.thunk(_THUNKS[$index], _rb_library(), ",
                repr(String(row["symbol"])), "), $guarded, _slots, _ret_p)")
        finish("        ", result)
        println(io, "    end")
    end
    println(io, "end")
    return Emitted(name, String(take!(io)), tier, "")
end

function _stub(name::String, path::String, reason::String)::String
    return "\"\"\"\n    $name(args...)\n\nNot wrapped: $(replace(reason, "\"\"\"" => "\"")).\n\nRust: `$path`\n\"\"\"\n" *
           "$name(args...) = Base.error($(repr("$name is not wrapped: $reason (Rust: $path)")))\n"
end

# =============================================================================
# Types as text
# =============================================================================

# The byte buffer a value of `size` bytes is built in, and a pointer to it.
_buffer(size::Int) = "Base.RefValue{Base.NTuple{$size,Base.UInt8}}(Base.ntuple(_ -> 0x00, Base.Val($size)))"
_buffer_ptr(size::Int, b::String) = "Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{Base.NTuple{$size,Base.UInt8}}, $b))"

# The fields of a record a property can read where the record is, and the
# variants of an enum: (name, discriminant value or :default, payload type name).
function _readable_fields(ctx::Ctx, t::AbstractDict)
    fields = Tuple{String,String}[]
    for f in get(t, "fields", Any[])
        (f["offset"] isa Integer && f["type"] isa AbstractString) || continue
        d = decode(ctx, f["type"], "_p", Int(f["offset"]); take = false)
        d === nothing || push!(fields, (String(f["name"]), d))
    end
    return fields
end

function _variants(ctx::Ctx, t::AbstractDict)
    variants = Tuple{String,Any,String}[]
    for v in get(t, "variants", Any[])
        haskey(v, "name") || continue
        vt = get(v, "type", nothing)
        vn = vt isa AbstractString ? get(ctx.names, vt, "") : ""
        push!(variants, (String(v["name"]), get(v, "default", false) === true ? :default : get(v, "discr_value", nothing), vn))
    end
    return variants
end

# `RustRuntime.variant` for an enum that carries data; `address` is the expression for the value's place.
function _emit_variant(io::IO, ctx::Ctx, t::AbstractDict, name::String, variants, open::String, address::String)
    place = _discriminant(ctx, t)
    lits = [v[2] === :default ? :default : (place === nothing ? nothing : _discr_literal(v[2], place.size)) for v in variants]
    (place === nothing || any(isnothing, lits)) && return
    println(io, "function RepliBuild.RustRuntime.variant(x::$name)")
    println(io, "    ", open)
    println(io, "    _d = GC.@preserve _r ", _load("Base." * _UINT[place.size], address, place.offset))
    default = :unknown
    for ((vname, _, _), lit) in zip(variants, lits)
        lit === :default ? (default = Symbol(vname)) : println(io, "    _d == $lit && return $(_sym(vname))")
    end
    println(io, "    return $(repr(default))")
    println(io, "end")
end

# A type Rust must drop: a Julia object that holds the value's bytes and
# drops it when it is collected.
function _emit_owned(io::IO, ctx::Ctx, key::String, t::AbstractDict, c::T)
    name = ctx.names[key]
    size = c.size
    drop = c.extra.drop
    kind = String(t["kind"])
    println(io, "\"\"\"")
    println(io, "Rust `$key` ($size bytes), held by this object. Rust's own drop code for it runs when Julia")
    println(io, "collects the object, or at `RepliBuild.RustRuntime.drop!`. Passing it to a Rust function by")
    println(io, "value moves it: the object is empty afterwards (`RepliBuild.RustRuntime.isowned`).")
    println(io, "\"\"\"")
    println(io, "mutable struct $name")
    println(io, "    _rb_bytes::Base.NTuple{$size,Base.UInt8}")
    println(io, "    _rb_owned::Base.Bool")
    println(io, "    function $name(::_RbRaw, bytes::Base.NTuple{$size,Base.UInt8})")
    println(io, "        x = new(bytes, true)")
    println(io, "        Base.finalizer(RepliBuild.RustRuntime.drop!, x)")
    println(io, "        return x")
    println(io, "    end")
    println(io, "end")
    println(io, "function RepliBuild.RustRuntime.drop!(x::$name)")
    println(io, "    if Base.getfield(x, :_rb_owned)")
    println(io, "        Base.setfield!(x, :_rb_owned, false)")
    println(io, "        GC.@preserve x $(_helper(drop))(Base.Ptr{Base.Cvoid}(Base.pointer_from_objref(x)))")
    println(io, "    end")
    println(io, "    return nothing")
    println(io, "end")
    println(io, "RepliBuild.RustRuntime.isowned(x::$name) = Base.getfield(x, :_rb_owned)")
    println(io, "function _rb_address(x::$name)")
    println(io, "    Base.getfield(x, :_rb_owned) || Base.error(\"this $name was moved into Rust, or dropped\")")
    println(io, "    return Base.Ptr{Base.Cvoid}(Base.pointer_from_objref(x))")
    println(io, "end")
    println(io, "Base.show(io::Base.IO, x::$name) = Base.print(io, \"$name(\", ",
            "Base.getfield(x, :_rb_owned) ? \"a Rust value\" : \"moved or dropped\", \")\")")
    make(body) = begin
        println(io, "    _b = ", _buffer(size))
        println(io, "    GC.@preserve _b ", body)
        println(io, "    return $name(_RbRaw(), _b[])")
        println(io, "end")
    end
    if c.kind === :string
        println(io, "function $name(s::Base.AbstractString)")
        make("_rb_string_new!($(_buffer_ptr(size, "_b")), s)")
        println(io, "Base.String(x::$name) = GC.@preserve x _rb_string_at(_rb_address(x))")
    elseif c.kind === :vec
        if _vec_buildable(ctx, c)
            println(io, "function $name(v::Base.AbstractVector)")
            make("$(_helper(c.extra.new))!($(_buffer_ptr(size, "_b")), v)")
        end
        println(io, "Base.length(x::$name) = GC.@preserve x $(_helper(c.extra.view))(_rb_address(x))[2]")
        read = decode(ctx, key, "_p", 0; take = false)
        read === nothing ||
            println(io, "Base.collect(x::$name) = GC.@preserve x (let _p = _rb_address(x); $read end)")
    elseif c.extra.box isa AbstractString
        println(io, "_rb_target(x::$name) = GC.@preserve x Base.unsafe_load(Base.Ptr{Base.Ptr{Base.Cvoid}}(_rb_address(x)))")
        read = decode(ctx, c.extra.box, "_q", 0; take = false)
        read === nothing ||
            println(io, "Base.getindex(x::$name) = GC.@preserve x (let _q = _rb_target(x); $read end)")
    elseif kind in ("struct", "enum", "union") && !(startswith(key, "alloc::") || startswith(key, "core::") || startswith(key, "std::"))
        # The crate's own type: its fields are read where the value is.
        fields = _readable_fields(ctx, t)
        if !isempty(fields)
            println(io, "function Base.getproperty(x::$name, f::Base.Symbol)")
            println(io, "    (f === :_rb_bytes || f === :_rb_owned) && return Base.getfield(x, f)")
            println(io, "    GC.@preserve x begin")
            println(io, "        _p = _rb_address(x)")
            for (fname, d) in fields
                println(io, "        f === $(_sym(fname)) && return $d")
            end
            println(io, "    end")
            println(io, "    Base.error(\"type $name has no field \$f\")")
            println(io, "end")
            println(io, "Base.propertynames(::$name) = (", join((_sym(f[1]) for f in fields), ", "), ",)")
        end
        kind == "enum" && _emit_variant(io, ctx, t, name, _variants(ctx, t), "_r = x", "_rb_address(x)")
    end
    println(io)
end

function _emit_types(ctx::Ctx)::String
    io = IOBuffer()
    for key in sort!(collect(keys(ctx.names)); by = k -> ctx.names[k])
        t = ctx.types[key]
        name = ctx.names[key]
        kind = String(t["kind"])
        size = get(t, "size", nothing)
        get(t, "conflict", false) === true && continue
        c = classify(ctx, key)
        if c.kind in (:owned, :string, :vec)
            _emit_owned(io, ctx, key, t, c)
            continue
        end
        kind == "pointer" && continue
        size isa Integer || continue
        if kind == "c_enum"
            haskey(_UINT, size) || continue
            members = String[]
            seen = Set{Any}()
            base = _enum_base(ctx.types, t)
            for e in get(t, "enumerators", Any[])
                e[2] isa Integer || continue
                v = _enum_value(e[2], base, Int(size))
                v in seen && continue                                  # Julia's @enum refuses a value twice
                push!(seen, v)
                push!(members, "$(name)_$(sanitize(String(e[1]))) = $v")
            end
            isempty(members) && continue
            println(io, "\"Rust enum `$key`.\"")
            println(io, "@enum $name::$base begin")
            foreach(m -> println(io, "    ", m), members)
            println(io, "end\n")
            continue
        end
        println(io, "\"\"\"")
        println(io, "Rust `$key` ($size bytes), as its bytes; its fields are properties read at the offsets the compiler chose.")
        println(io, "\"\"\"")
        println(io, "struct $name")
        println(io, "    _rb_bytes::Base.NTuple{$size,Base.UInt8}")
        println(io, "    $name(::_RbRaw, bytes::Base.NTuple{$size,Base.UInt8}) = new(bytes)")
        println(io, "end")
        fields = _readable_fields(ctx, t)
        encodable = unmovable(ctx, key) === nothing
        ctor = String[]
        for (i, f) in enumerate(get(t, "fields", Any[]))
            (f["offset"] isa Integer && f["type"] isa AbstractString) || (encodable = false; continue)
            # A record is built from Julia only out of plain parts: one that
            # owns something is made by the library.
            fk = classify(ctx, f["type"]).kind
            e = fk in (:owned, :string, :vec) ? nothing : encode(ctx, f["type"], "_p", Int(f["offset"]), "f$i")
            e === nothing ? (encodable = false) : append!(ctor, e)
        end
        variants = kind == "enum" ? _variants(ctx, t) : Tuple{String,Any,String}[]
        if size > 0 && (!isempty(fields) || !isempty(variants))
            println(io, "function Base.getproperty(x::$name, f::Base.Symbol)")
            println(io, "    f === :_rb_bytes && return Base.getfield(x, :_rb_bytes)")
            println(io, "    _r = Base.RefValue{$name}(x)")
            println(io, "    GC.@preserve _r begin")
            println(io, "        _p = Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{$name}, _r))")
            for (fname, d) in fields
                println(io, "        f === $(_sym(fname)) && return $d")
            end
            for (vname, _, vn) in variants
                isempty(vn) && continue
                println(io, "        f === $(_sym(vname)) && return (RepliBuild.RustRuntime.variant(x) === $(_sym(vname)) ? ",
                        _load(vn, "_p", 0), " : Base.error(\"this $name is not a $vname\"))")
            end
            println(io, "    end")
            println(io, "    Base.error(\"type $name has no field \$f\")")
            println(io, "end")
            props = vcat([f[1] for f in fields], [v[1] for v in variants if !isempty(v[3])])
            println(io, "Base.propertynames(::$name) = (", join((_sym(p) for p in props), ", "), isempty(props) ? "" : ",", ")")
        end
        if kind == "enum"
            _emit_variant(io, ctx, t, name, variants, "_r = Base.RefValue{$name}(x)",
                          "Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{$name}, _r))")
        elseif kind == "struct" && encodable && size > 0 && !isempty(get(t, "fields", Any[]))
            n = length(t["fields"])
            println(io, "function $name(", join(("f$i" for i in 1:n), ", "), ")")
            println(io, "    _b = ", _buffer(size))
            println(io, "    GC.@preserve _b begin")
            println(io, "        _p = ", _buffer_ptr(size, "_b"))
            foreach(s -> println(io, "        ", s), ctor)
            println(io, "    end")
            println(io, "    return $name(_RbRaw(), _b[])")
            println(io, "end")
        end
        println(io)
    end
    return String(take!(io))
end

"""
    _emit_ownership(ctx) -> String

The module's calls into the root crate's own functions: one that drops a
value per owning type, the two that read and build a `String`, and per `Vec`
type the two that read its elements and build one. Each is a plain `ccall`:
those functions are RepliBuild's, written with C's calling convention.
"""
function _emit_ownership(ctx::Ctx)::String
    io = IOBuffer()
    owned = [(key, classify(ctx, key)) for key in sort!(collect(keys(ctx.glue)))]
    filter!(kc -> kc[2].kind in (:owned, :string, :vec), owned)
    isempty(owned) && return ""
    println(io, "# The object gives up its Rust value: whoever the bytes go to drops it from here on.")
    println(io, "function _rb_take!(x)")
    println(io, "    Base.getfield(x, :_rb_owned) ||")
    println(io, "        Base.error(\"this \$(Base.nameof(Base.typeof(x))) was already moved into Rust, or dropped\")")
    println(io, "    Base.setfield!(x, :_rb_owned, false)")
    println(io, "    return Base.getfield(x, :_rb_bytes)")
    println(io, "end")
    for (key, c) in owned
        drop = c.extra.drop
        # A type that was not spelled: the library's static holds where the
        # drop function the compiler picked for it is.
        callee = get(ctx.glue[key], "by_address", false) === true ?
            "Base.unsafe_load(Base.cglobal(($(_sym(drop)), LIBRARY_PATH), Base.Ptr{Base.Cvoid}))" :
            "($(_sym(drop)), LIBRARY_PATH)"
        println(io, "$(_helper(drop))(p::Base.Ptr{Base.Cvoid}) = ccall($callee, Base.UInt, ",
                "(Base.Ptr{Base.Cvoid}, Base.Ptr{Base.Cvoid}), p, Base.C_NULL)")
    end
    for (key, c) in owned
        bytes = "Base.NTuple{$(c.size),Base.UInt8}"
        if c.kind === :string
            name = c.extra.name
            println(io)
            println(io, "# Rust's `String`: read and built by the library's own code.")
            println(io, "function _rb_string_at(p::Base.Ptr{Base.Cvoid})")
            println(io, "    d = Base.RefValue{Base.Ptr{Base.UInt8}}(Base.C_NULL)")
            println(io, "    n = Base.RefValue{Base.UInt}(0)")
            println(io, "    ccall((:__rb_string_view, LIBRARY_PATH), Base.Cvoid, ",
                    "(Base.Ptr{Base.Cvoid}, Base.Ptr{Base.Ptr{Base.UInt8}}, Base.Ptr{Base.UInt}), p, d, n)")
            println(io, "    return n[] == 0 ? \"\" : Base.unsafe_string(d[], n[])")
            println(io, "end")
            println(io, "function _rb_string_take!(p::Base.Ptr{Base.Cvoid})")
            println(io, "    s = _rb_string_at(p)")
            println(io, "    $(_helper(c.extra.drop))(p)")
            println(io, "    return s")
            println(io, "end")
            println(io, "function _rb_string_new!(p::Base.Ptr{Base.Cvoid}, s::Base.AbstractString)")
            println(io, "    t = Base.String(s)")
            println(io, "    ok = GC.@preserve t ccall((:__rb_string_new, LIBRARY_PATH), Base.UInt8, ",
                    "(Base.Ptr{Base.UInt8}, Base.UInt, Base.Ptr{Base.Cvoid}), Base.pointer(t), Base.ncodeunits(t), p)")
            println(io, "    ok == 0x00 || Base.throw(Base.ArgumentError(\"a Rust `String` must be valid UTF-8\"))")
            println(io, "    return nothing")
            println(io, "end")
            println(io, "_rb_string_new!(p::Base.Ptr{Base.Cvoid}, x::$name) = (", _store(bytes, "p", 0, "_rb_take!(x)"), "; nothing)")
        elseif c.kind === :vec
            e = c.extra
            println(io)
            println(io, "# Rust's `$key`.")
            println(io, "function $(_helper(e.view))(p::Base.Ptr{Base.Cvoid})")
            println(io, "    d = Base.RefValue{Base.Ptr{Base.Cvoid}}(Base.C_NULL)")
            println(io, "    n = Base.RefValue{Base.UInt}(0)")
            println(io, "    ccall(($(_sym(e.view)), LIBRARY_PATH), Base.Cvoid, ",
                    "(Base.Ptr{Base.Cvoid}, Base.Ptr{Base.Ptr{Base.Cvoid}}, Base.Ptr{Base.UInt}), p, d, n)")
            println(io, "    return (d[], Base.Int(n[]))")
            println(io, "end")
            enc = encode(ctx, e.elem, "_e", 0, "_x")
            if enc !== nothing
                println(io, "function $(_helper(e.new))!(p::Base.Ptr{Base.Cvoid}, v::Base.AbstractVector)")
                println(io, "    n = Base.length(v)")
                println(io, "    buf = Base.zeros(Base.UInt8, n * $(e.stride))")
                println(io, "    GC.@preserve buf begin")
                println(io, "        _b = Base.Ptr{Base.Cvoid}(Base.pointer(buf))")
                println(io, "        for (_i, _x) in Base.enumerate(v)")
                println(io, "            _e = _b + (_i - 1) * $(e.stride)")
                foreach(s -> println(io, "            ", s), enc)
                println(io, "        end")
                println(io, "        ccall(($(_sym(e.new)), LIBRARY_PATH), Base.Cvoid, ",
                        "(Base.Ptr{Base.UInt8}, Base.UInt, Base.Ptr{Base.Cvoid}), buf, n, p)")
                println(io, "    end")
                println(io, "    return nothing")
                println(io, "end")
                println(io, "$(_helper(e.new))!(p::Base.Ptr{Base.Cvoid}, x::$(e.name)) = (", _store(bytes, "p", 0, "_rb_take!(x)"), "; nothing)")
            end
        end
    end
    println(io)
    return String(take!(io))
end

# =============================================================================
# The module
# =============================================================================

"""
    generate(metadata; module_name, library_path, generator = "") -> (text, report)

The wrapper module for a Rust library, from its build's metadata (parsed).
`report` lists, per function, its Julia name and how it calls
(`"ccall"`, `"thunk"`, `"guarded"`, `"unsupported"` with a reason).

The text is a function of the metadata and the names given: rows are taken in
the metadata's order (path, then symbol), types by name.
"""
function generate(metadata::AbstractDict; module_name::AbstractString, library_path::AbstractString,
                  generator::AbstractString = "")
    get(metadata, "language", "") == "rust" || error("GeneratorRust.generate: the metadata is not a Rust library's")
    ctx = Ctx(metadata)
    type_names = Set(values(ctx.names))

    rows = metadata["functions"]
    # Names: a function's own, then its scope in front where two share one.
    wanted = [function_name(r) for r in rows]
    counts = Dict{String,Int}()
    foreach(n -> counts[n] = get(counts, n, 0) + 1, wanted)
    taken = Set{String}()
    names = String[]
    for (r, n) in zip(rows, wanted)
        name = n
        if counts[n] > 1
            scope = [sanitize(String(s)) for s in get(r, "scope", Any[])]
            length(scope) > 1 && (name = join(scope[2:end], "_") * "_" * n)
        end
        (name in type_names || _reserved(name) || name == module_name) && (name *= "_")
        base = name
        k = 2
        while name in taken
            name = base * "_" * string(k)
            k += 1
        end
        push!(taken, name)
        push!(names, name)
    end

    emitted = Emitted[emit_function(ctx, r, n, i) for (i, (r, n)) in enumerate(zip(rows, names))]
    report = [Dict{String,Any}("name" => e.name, "symbol" => String(r["symbol"]), "path" => String(r["path"]),
                               "tier" => e.tier, "reason" => e.reason) for (e, r) in zip(emitted, rows)]

    io = IOBuffer()
    println(io, "# Auto-generated Julia wrapper for the Rust crate `$(metadata["crate"]["name"])`")
    println(io, "# Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println(io, "# Generator: RepliBuild (Rust front end)", isempty(generator) ? "" : " $generator")
    println(io, "# Library: $(basename(library_path))")
    println(io, "# Metadata: compilation_metadata.json")
    println(io)
    println(io, "module $module_name")
    println(io)
    println(io, "import RepliBuild")
    println(io)
    println(io, "# The library beside this file first: a build's files travel together.")
    println(io, "const LIBRARY_PATH = let baked = $(repr(String(library_path)))")
    println(io, "    sibling = Base.joinpath(Base.@__DIR__, Base.basename(baked))")
    println(io, "    Base.isfile(sibling) ? sibling : baked")
    println(io, "end")
    println(io, "Base.isfile(LIBRARY_PATH) || Base.error(\"Library not found: \$LIBRARY_PATH\")")
    println(io)
    println(io, "# This wrapper's offsets and calls are those of ONE build of the library.")
    println(io, "const LIBRARY_SHA = $(repr(String(metadata["library_sha"])))")
    println(io, "const BUILD_GENERATOR = $(repr(String(generator)))")
    println(io, "const RUST_TOOLCHAIN = $(repr(String(metadata["toolchain"]["rustc"])))")
    println(io)
    println(io, "struct _RbRaw end")
    println(io, "_rb_address(x::Base.Ptr) = Base.Ptr{Base.Cvoid}(x)")
    println(io, "_rb_address(x::Base.RefValue{T}) where {T} = Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{T}, x))")
    println(io)
    println(io, "const _LIBRARY = Base.RefValue{Base.Union{Base.Nothing,RepliBuild.RustRuntime.Library}}(nothing)")
    println(io, "function _rb_library()")
    println(io, "    lib = _LIBRARY[]")
    println(io, "    lib === nothing || return lib")
    println(io, "    return _LIBRARY[] = RepliBuild.RustRuntime.Library(LIBRARY_PATH)")
    println(io, "end")
    println(io, "const _THUNKS = [Base.RefValue{Base.Ptr{Base.Cvoid}}(Base.C_NULL) for _ in 1:$(length(rows))]")
    println(io)
    println(io, "function __init__()")
    println(io, "    _LIBRARY[] = nothing")
    println(io, "    for cell in _THUNKS")
    println(io, "        cell[] = Base.C_NULL")
    println(io, "    end")
    println(io, "    sha = Base.bytes2hex(Base.open(RepliBuild.SHA.sha256, LIBRARY_PATH))")
    println(io, "    sha == LIBRARY_SHA || @warn \"$module_name: \$LIBRARY_PATH is not the library this wrapper was generated from; \" *")
    println(io, "                                \"its offsets and calls may be wrong. Run RepliBuild.wrap again.\"")
    println(io, "    return nothing")
    println(io, "end")
    println(io)
    println(io, "# ── Types ────────────────────────────────────────────────────────────────")
    println(io)
    print(io, _emit_types(ctx))
    ownership = _emit_ownership(ctx)
    if !isempty(ownership)
        println(io, "# ── What Rust must drop ──────────────────────────────────────────────────")
        println(io)
        print(io, ownership)
    end
    println(io, "# ── Functions ────────────────────────────────────────────────────────────")
    println(io)
    for e in emitted
        println(io, e.text)
    end
    println(io, "# ── What is in this module ───────────────────────────────────────────────")
    println(io)
    println(io, "\"Each wrapped function's Julia name, and how it calls: `:ccall`, `:thunk` or `:guarded` (a thunk inside the frame that catches a panic).\"")
    println(io, "const FUNCTIONS = Base.Dict{Base.Symbol,Base.Symbol}(")
    for e in emitted
        e.tier == "unsupported" || println(io, "    $(_sym(e.name)) => $(_sym(e.tier)),")
    end
    println(io, ")")
    println(io)
    println(io, "\"Each function of the library that is NOT wrapped, and why.\"")
    println(io, "const UNSUPPORTED = Base.Dict{Base.Symbol,Base.String}(")
    for e in emitted
        e.tier == "unsupported" && println(io, "    $(_sym(e.name)) => $(repr(e.reason)),")
    end
    println(io, ")")
    println(io)
    println(io, "\"Each type of this module that holds a Rust value Rust must drop, and the Rust type it is.\"")
    println(io, "const OWNED = Base.Dict{Base.Symbol,Base.String}(")
    for key in sort!(collect(keys(ctx.names)); by = k -> ctx.names[k])
        classify(ctx, key).kind in (:owned, :string, :vec) && println(io, "    $(_sym(ctx.names[key])) => $(repr(String(key))),")
    end
    println(io, ")")
    println(io)
    println(io, "end # module $module_name")
    return (String(take!(io)), report)
end

# A wrapper that does not parse is this generator's bug, and is never written.
function _assert_parses(text::AbstractString, where::AbstractString)
    ex = Meta.parseall(text; filename = where)
    for arg in ex.args
        (arg isa Expr && arg.head in (:incomplete, :error)) &&
            error("GeneratorRust wrote a wrapper that does not parse ($where): $(arg.args)")
    end
    return nothing
end

"""
    wrap_project(config) -> String

`RepliBuild.wrap` for a `[rust]` project: read the build's metadata, write
`<[paths] output>/<Module>.jl`, and say what was wrapped and what was not.
Returns the wrapper's path.
"""
function wrap_project(config::RepliBuildConfig)::String
    out_dir = get_output_path(config)
    library = joinpath(out_dir, get_library_name(config))
    meta = joinpath(out_dir, "compilation_metadata.json")
    isfile(library) || error("Library not found: $library\nRun RepliBuild.build first.")
    isfile(meta) || error("$meta is missing: RepliBuild.build writes it, and the wrapper is generated from it.")
    metadata = _read_json(meta)
    module_name = get_module_name(config)
    version = something(pkgversion(@__MODULE__), "")
    (text, report) = generate(metadata; module_name, library_path = abspath(library), generator = string(version))
    path = joinpath(out_dir, module_name * ".jl")
    _assert_parses(text, path)
    write(path, text)
    tiers = Dict{String,Int}()
    foreach(r -> tiers[r["tier"]] = get(tiers, r["tier"], 0) + 1, report)
    println("RepliBuild | $(config.project.name) [rust] wrap")
    println("  wrapper: $(basename(path)): $(length(report) - get(tiers, "unsupported", 0)) of $(length(report)) functions " *
            "($(get(tiers, "ccall", 0)) plain calls, $(get(tiers, "thunk", 0) + get(tiers, "guarded", 0)) through thunks, " *
            "$(get(tiers, "guarded", 0)) of those may panic)")
    reasons = Dict{String,Vector{String}}()
    for r in report
        r["tier"] == "unsupported" && push!(get!(reasons, replace(String(r["reason"]), r"`[^`]*`" => "…"), String[]), String(r["name"]))
    end
    for reason in sort!(collect(keys(reasons)); by = k -> (-length(reasons[k]), k))
        ns = sort!(reasons[reason])
        println("  not wrapped ($(length(ns))): $reason: ", join(ns[1:min(end, 3)], ", "), length(ns) > 3 ? ", and $(length(ns) - 3) more" : "")
    end
    return path
end

end # module GeneratorRust
