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
# What this does NOT wrap yet is a function that moves a value which may own
# memory (`String`, `Vec`, `Box`, anything holding one): Julia would have to
# free it with Rust's own drop code, which is not built. Such a function is
# emitted as a stub that says so, and listed in the module's `UNSUPPORTED`.
# Nothing is guessed, and nothing is silently left out.
#
# Design: docs/updates/RUST_FRONT_END.md.

module GeneratorRust

using Dates

import ..ConfigurationManager
import ..ConfigurationManager: RepliBuildConfig, get_output_path, get_library_name, get_module_name
import ..RustABI
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
    kind::Symbol                 # :int :float :bool :char :unit :cenum :blob :tuple :str :slice :dyn :ptr :option :unknown
    jl::String                   # the Julia type a value of it has (for :ptr, the pointee's, or "")
    size::Int
    key::String
    extra::Any
end

mutable struct Ctx
    types::Dict{String,Any}
    names::Dict{String,String}   # type key => generated Julia type name
    memo::Dict{String,T}
end

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
        jl = ""
        if target isa AbstractString
            tt = classify(ctx, target)
            tt.kind in (:int, :float, :bool, :cenum, :blob) && (jl = tt.jl)
        end
        return T(:ptr, jl, something(size, RustABI.POINTER_BYTES), key, pk)
    elseif kind == "c_enum"
        (size isa Integer && haskey(_UINT, size)) || return T(:unknown, "", -1, key, "the enum `$key` has no size")
        return T(:cenum, ctx.names[key], size, key, "Base." * _UINT[size])
    elseif kind in ("struct", "enum", "union")
        size isa Integer || return T(:unknown, "", -1, key, "the type `$key` has no size")
        fat = String(get(t, "fat", ""))
        fields = get(t, "fields", Any[])
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
        return T(:blob, ctx.names[key], size, key, kind)
    end
    return T(:unknown, "", -1, key, "the type `$key` is not one this wraps")
end

# `Option<P>`: (payload key, payload offset, :tag or :null, tag type key, tag offset, none value, some value).
function _option(ctx::Ctx, t::AbstractDict)
    (startswith(String(get(t, "name", "")), "Option<") && get(t, "scope", Any[]) == ["core", "option"]) || return nothing
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
        # A niche: `None` is one value of the payload's own bytes. Only the
        # null pointer is read here.
        pk = get(get(ctx.types, payload, Dict()), "kind", "")
        (pk == "pointer" && nv == 0 && f[1]["offset"] == d["offset"]) || return nothing
        return (payload = payload, offset = Int(f[1]["offset"]), mode = :null)
    end
    sv_value = get(variants[some], "discr_value", nothing)
    (nv isa Integer && sv_value isa Integer && d["type"] isa AbstractString && d["offset"] isa Integer) || return nothing
    return (payload = payload, offset = Int(f[1]["offset"]), mode = :tag, tag = String(d["type"]),
            tag_offset = Int(d["offset"]), none = Int(nv), some = Int(sv_value))
end

"""
    owns(ctx, key) -> Bool

Whether a value of this type, held by value, may own memory: a `Box`, or any
pointer inside a record that is not a plain reference or a fat pointer, or
anything of `alloc`. Conservative: a record with a raw pointer field counts.
A function that moves such a value needs Rust's drop code on the Julia side,
which is not built, so it is not wrapped.
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

# A buffer Julia allocates is aligned to 16 bytes. A type Rust aligns more
# strictly cannot be built in one or returned into one.
_overaligned(ctx::Ctx, key) = key isa AbstractString && something(get(get(ctx.types, key, Dict()), "align", nothing), 1) > 16

# The Julia names of the types the module defines: records, enums, unions and
# fieldless enums. A variant's record is named after its enum (`Shape_Circle`).
function _type_names(types::AbstractDict)::Dict{String,String}
    names = Dict{String,String}()
    taken = Set{String}()
    for key in sort!(collect(keys(types)))
        t = types[key]
        kind = get(t, "kind", "")
        kind in ("struct", "enum", "union", "c_enum") || continue
        (isempty(get(t, "fat", "")) && get(t, "tuple", false) !== true) || continue
        scope = String[String(x) for x in get(t, "scope", Any[])]
        parent = join(scope, "::")
        base = sanitize(String(get(t, "name", key)))
        haskey(types, parent) && get(types[parent], "kind", "") == "enum" &&
            (base = sanitize(String(get(types[parent], "name", parent))) * "_" * base)
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

"The expression that reads a value of type `key` from memory at `p + off`, or `nothing` when this cannot."
function decode(ctx::Ctx, key::Union{AbstractString,Nothing}, p::String, off::Int)::Union{String,Nothing}
    t = classify(ctx, key)
    k = t.kind
    k === :unit && return "nothing"
    k in (:int, :float) && return _load(t.jl, p, off)
    k === :bool && return "(" * _load("Base.UInt8", p, off) * " != 0x00)"
    k === :char && return "Base.Char(" * _load("Base.UInt32", p, off) * ")"
    k === :cenum && return "$(t.jl)(" * _load(t.extra, p, off) * ")"
    k === :blob && return _load(t.jl, p, off)
    if k === :ptr
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
            d = decode(ctx, ek, p, off + eo)
            d === nothing && return nothing
            push!(parts, d)
        end
        return "(" * join(parts, ", ") * (length(parts) == 1 ? "," : "") * ")"
    elseif k === :option
        o = t.extra
        inner = decode(ctx, o.payload, p, off + o.offset)
        inner === nothing && return nothing
        if o.mode === :null
            return "(" * _load("Base.UInt", p, off + o.offset) * " == 0 ? nothing : $inner)"
        end
        tag = classify(ctx, o.tag)
        tag.kind === :int || return nothing
        return "(" * _load(tag.jl, p, off + o.tag_offset) * " == $(o.some) ? $inner : nothing)"
    end
    return nothing
end

"The statements that write `v` as a value of type `key` to memory at `p + off`, or `nothing`."
function encode(ctx::Ctx, key::Union{AbstractString,Nothing}, p::String, off::Int, v::String)::Union{Vector{String},Nothing}
    t = classify(ctx, key)
    k = t.kind
    k === :unit && return String[]
    k in (:int, :float) && return [_store(t.jl, p, off, "Base.convert($(t.jl), $v)")]
    k === :bool && return [_store("Base.UInt8", p, off, "Base.UInt8($v::Base.Bool)")]
    k === :char && return [_store("Base.UInt32", p, off, "Base.UInt32(Base.Char($v))")]
    k === :cenum && return [_store(t.extra, p, off, "$(t.extra)(Base.Integer($v::$(t.jl)))")]
    k === :blob && return [_store(t.jl, p, off, "$v::$(t.jl)")]
    if k === :tuple
        out = String[]
        for (i, (eo, ek)) in enumerate(t.extra)
            e = encode(ctx, ek, p, off + eo, "$v[$i]")
            e === nothing && return nothing
            append!(out, e)
        end
        return out
    elseif k === :option
        o = t.extra
        if o.mode === :tag
            tag = classify(ctx, o.tag)
            tag.kind === :int || return nothing
            inner = encode(ctx, o.payload, p, off + o.offset, v)
            inner === nothing && return nothing
            return ["if $v === nothing",
                    "    " * _store(tag.jl, p, off + o.tag_offset, "Base.convert($(tag.jl), $(o.none))"),
                    "else",
                    "    " * _store(tag.jl, p, off + o.tag_offset, "Base.convert($(tag.jl), $(o.some))"),
                    ["    " * s for s in inner]...,
                    "end"]
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
    k in (:cenum, :blob) && return t.jl
    k === :tuple && return "Base.Tuple"
    k === :str && return "Base.AbstractString"
    k === :slice && return "Base.DenseVector{$(t.jl)}"
    if k === :option
        o = t.extra
        o.mode === :null && return "Base.Union{Base.Nothing,Base.RefValue,Base.Ptr}"
        return "Base.Union{Base.Nothing,$(julia_arg_type(ctx, o.payload))}"
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
    ir = row["ir"]
    irp = ir["parameters"]
    ret = row["return"]
    rpass = String(get(ret, "pass", ""))

    prelude = String[]            # statements before the call
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
                                      "$jt(Base.Integer($pname))"
            # A scalar wider or narrower than its IR type is not a direct scalar this reads.
            (t.kind === :int && t.size != something(RustABI._ir_scalar_bytes(irt[1]), -1)) &&
                return unsupported("the parameter `$shown` is passed as `$(irt[1])`, not at its own width")
            values[irs[1]] = (jt, expr)
            push!(args, "$pname::$(julia_arg_type(ctx, p["type"]))")
        elseif t.kind in (:str, :slice) && pass == "pair"
            if t.kind === :str
                push!(prelude, "$(pname)_s = Base.String($pname)")
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
            t.extra == "box" && return unsupported("the parameter `$shown` is a `Box`, which owns its value; ownership is not built yet")
            if t.extra == "ref" && !isempty(t.jl)
                push!(prelude, "$(pname)_r = $pname isa Base.Ptr ? $pname : $pname isa Base.RefValue ? $pname : Base.RefValue{$(t.jl)}($pname)")
                push!(args, "$pname::Base.Union{$(t.jl),Base.RefValue{$(t.jl)},Base.Ptr}")
            else
                push!(prelude, "$(pname)_r = $pname")
                push!(args, "$pname::Base.Union{Base.RefValue,Base.Ptr}")
            end
            push!(keep, "$(pname)_r")
            values[irs[1]] = ("Base.Ptr{Base.Cvoid}", "_rb_address($(pname)_r)")
        elseif t.kind === :option && t.extra.mode === :null && pass == "direct"
            push!(prelude, "$(pname)_r = $pname")
            push!(keep, "$(pname)_r")
            values[irs[1]] = ("Base.Ptr{Base.Cvoid}", "($(pname)_r === nothing ? Base.C_NULL : _rb_address($(pname)_r))")
            push!(args, "$pname::$(julia_arg_type(ctx, p["type"]))")
        elseif t.kind in (:blob, :tuple, :option, :int, :float, :bool, :char, :cenum)
            # By value, as its bytes; the IR arguments are read from them.
            owns(ctx, p["type"]) &&
                return unsupported("the parameter `$shown` moves a `$(p["type"])`, which may own memory; ownership is not built yet")
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
        (rt.kind === :ptr && rt.extra == "box") &&
            return unsupported("it returns a `Box`, which owns its value; ownership is not built yet")
        rt.kind in (:str, :slice, :ptr) || !owns(ctx, ret["type"]) ||
            return unsupported("it returns a `$(ret["type"])`, which may own memory; ownership is not built yet")
        _overaligned(ctx, ret["type"]) && return unsupported("it returns a type aligned beyond 16 bytes")
        rsize = rt.size
        result = decode(ctx, ret["type"], "_ret_p", 0)
        result === nothing && return unsupported("a `$(ret["type"])` cannot be read back yet")
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
                " (merged with another function by the compiler: the types shown are that function's)" : "")
    println(io, "\"\"\"")
    println(io, "function $name(", join(args, ", "), ")")
    for s in prelude
        println(io, "    ", s)
    end
    vals = [values[i] for i in start:length(irp)]
    if plain
        types = "(" * join((v[1] for v in vals), ", ") * (length(vals) == 1 ? "," : "") * ")"
        rjl = ret_ir == "void" ? "Base.Cvoid" : ir_julia(ret_ir)
        call = "ccall(($(_sym(row["symbol"])), LIBRARY_PATH), $rjl, $types" *
               (isempty(vals) ? "" : ", " * join((v[2] for v in vals), ", ")) * ")"
        if rpass == "ignore"
            println(io, isempty(keep) ? "    $call" : "    GC.@preserve $(join(keep, " ")) $call")
            println(io, "    return nothing")
        else
            println(io, "    _ret = Base.RefValue{Base.NTuple{$(max(rsize, 1)),Base.UInt8}}()")
            println(io, "    _r = ", isempty(keep) ? call : "GC.@preserve $(join(keep, " ")) $call")
            println(io, "    GC.@preserve _ret begin")
            println(io, "        _ret_p = Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{Base.NTuple{$(max(rsize, 1)),Base.UInt8}}, _ret))")
            println(io, "        ", _store(rjl, "_ret_p", 0, "_r"))
            println(io, "        return $result")
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
        println(io, "        return $result")
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

function _emit_types(ctx::Ctx)::String
    io = IOBuffer()
    for key in sort!(collect(keys(ctx.names)); by = k -> ctx.names[k])
        t = ctx.types[key]
        name = ctx.names[key]
        kind = String(t["kind"])
        size = get(t, "size", nothing)
        get(t, "conflict", false) === true && continue
        size isa Integer || continue
        if kind == "c_enum"
            haskey(_UINT, size) || continue
            members = String[]
            seen = Set{Any}()
            for e in get(t, "enumerators", Any[])
                (e[2] isa Integer && !(e[2] in seen)) || continue      # Julia's @enum refuses a value twice
                push!(seen, e[2])
                push!(members, "$(name)_$(sanitize(String(e[1]))) = $(e[2])")
            end
            isempty(members) && continue
            println(io, "\"Rust enum `$key`.\"")
            println(io, "@enum $name::$(_UINT[size]) begin")
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
        fields = Tuple{String,String}[]          # (property, decode expression)
        encodable = true
        ctor = String[]
        for (i, f) in enumerate(get(t, "fields", Any[]))
            (f["offset"] isa Integer && f["type"] isa AbstractString) || (encodable = false; continue)
            d = decode(ctx, f["type"], "_p", Int(f["offset"]))
            e = encode(ctx, f["type"], "_p", Int(f["offset"]), "f$i")
            d === nothing || push!(fields, (String(f["name"]), d))
            e === nothing ? (encodable = false) : append!(ctor, e)
            owns(ctx, f["type"], 1, false) && (encodable = false)
        end
        variants = Tuple{String,Any,String}[]    # (name, discriminant value or :default, payload type name)
        if kind == "enum"
            for v in get(t, "variants", Any[])
                haskey(v, "name") || continue
                vt = get(v, "type", nothing)
                vn = vt isa AbstractString ? get(ctx.names, vt, "") : ""
                push!(variants, (String(v["name"]), get(v, "default", false) === true ? :default : get(v, "discr_value", nothing), vn))
            end
        end
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
            d = get(t, "discriminant", nothing)
            dt = d === nothing ? nothing : classify(ctx, get(d, "type", nothing))
            if d !== nothing && dt.kind === :int && d["offset"] isa Integer && all(v -> v[2] === :default || v[2] isa Integer, variants)
                println(io, "function RepliBuild.RustRuntime.variant(x::$name)")
                println(io, "    _r = Base.RefValue{$name}(x)")
                println(io, "    _d = GC.@preserve _r ", _load(dt.jl, "Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{$name}, _r))", Int(d["offset"])))
                default = :unknown
                for (vname, value, _) in variants
                    value === :default ? (default = Symbol(vname)) :
                        println(io, "    _d == $value && return $(_sym(vname))")
                end
                println(io, "    return $(repr(default))")
                println(io, "end")
            end
        elseif kind == "struct" && encodable && size > 0 && !isempty(get(t, "fields", Any[]))
            n = length(t["fields"])
            println(io, "function $name(", join(("f$i" for i in 1:n), ", "), ")")
            println(io, "    _b = Base.RefValue{Base.NTuple{$size,Base.UInt8}}(Base.ntuple(_ -> 0x00, Base.Val($size)))")
            println(io, "    GC.@preserve _b begin")
            println(io, "        _p = Base.Ptr{Base.Cvoid}(Base.unsafe_convert(Base.Ptr{Base.NTuple{$size,Base.UInt8}}, _b))")
            foreach(s -> println(io, "        ", s), ctor)
            println(io, "    end")
            println(io, "    return $name(_RbRaw(), _b[])")
            println(io, "end")
        end
        println(io)
    end
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
    types = metadata["types"]
    ctx = Ctx(types, _type_names(types), Dict{String,T}())
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
