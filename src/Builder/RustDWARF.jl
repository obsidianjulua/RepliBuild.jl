# RustDWARF.jl — what rustc's DWARF says about a crate's functions and types.
#
# rustc describes a build exactly: every record's fields at the offsets the
# compiler chose (it reorders them), every enum that carries data as a
# `DW_TAG_variant_part` with the discriminant's place and each variant's
# value, every pointer by its Rust spelling (`&T`, `&mut T`, `*const T`,
# `Box<T>`), every fat pointer as a two-field record. `Compiler.parse_dwarf_dump`
# reads the same text for C and C++, and its heuristics are the parts that are
# wrong here: it keys types by bare name (Rust has a `Some` per `Option<T>`),
# flattens a variant part into overlapping members, reduces every pointer to
# `T*`, and takes a method's class from an Itanium name. So Rust has its own
# pass, and this is it (docs/updates/RUST_FRONT_END.md, Part I §3.5).
#
# The input is the GNU binutils dump (`readelf --debug-dump=info`, or objdump's
# on PE, or `LibDW.DumpLines`, which writes the same lines), read one compile
# unit at a time. A Rust library carries the standard library's DWARF beside
# the crate's (436,491 lines for a 17-function crate, 1,498 of them the
# crate's), so a unit is turned into a DIE tree only when it names a function
# that was asked for, and a type is described only when one of those
# functions reaches it.
#
# Nothing here runs a tool: the lines come from the caller (`RustBuild`).

module RustDWARF

export Die, read_units, describe_functions

"One debugging information entry: its offset in `.debug_info`, tag and attributes as printed."
mutable struct Die
    offset::Int
    tag::String                          # `structure_type`, without `DW_TAG_`
    attrs::Dict{String,String}           # `byte_size` => `24`, without `DW_AT_`
    children::Vector{Die}
    parent::Union{Nothing,Die}
end
Die(offset::Int, tag::AbstractString, parent) =
    Die(offset, String(tag), Dict{String,String}(), Die[], parent)

"The compile units that were kept: each one's root DIE, and every kept DIE by offset."
struct Units
    roots::Vector{Die}
    by_offset::Dict{Int,Die}
end

const _DIE_LINE  = r"^\s*<(\d+)><([0-9a-f]+)>: Abbrev Number: (\d+)(?: \(DW_TAG_([A-Za-z0-9_]+)\))?"
const _ATTR_LINE = r"^\s*<[0-9a-f]+>\s+DW_AT_([A-Za-z0-9_]+)\s*:\s?(.*)$"
# `(indirect string, offset: 0x487e3): text`, `(indirect line string, …): text`,
# `(indexed string: 0x1): text`, `(alt indirect string, …): text`.
const _STRING_FORM = r"^\((?:indirect|indexed|alt indirect)[^)]*\):\s?"

_attr_value(v::AbstractString) = String(strip(replace(v, _STRING_FORM => ""; count = 1)))

"""
    read_units(lines, wanted) -> Units

Read a binutils `.debug_info` dump, one line per element of `lines`, and keep
the compile units that hold a `DW_AT_linkage_name` (or, for a symbol rustc
did not mangle, a `DW_AT_name`) in `wanted`. A unit that names none is never
built into a tree.

A unit starts at a DIE of depth 0; nothing else in the dump is read (the
`Compilation Unit @ offset` header and its fields are not DIEs).
"""
function read_units(lines, wanted::AbstractSet{<:AbstractString})::Units
    units = Units(Die[], Dict{Int,Die}())
    buf = String[]
    keep = false
    function flush!()
        (keep && !isempty(buf)) && _build_unit!(units, buf)
        empty!(buf)
        keep = false
    end
    for raw in lines
        line = String(raw)
        m = match(_DIE_LINE, line)
        if m !== nothing
            m.captures[1] == "0" && m.captures[3] != "0" && flush!()
            push!(buf, line)
            continue
        end
        isempty(buf) && continue                      # before the first DIE
        a = match(_ATTR_LINE, line)
        a === nothing && continue
        push!(buf, line)
        # A `#[no_mangle]` function's DIE has no linkage name: its name is the symbol.
        if !keep && (a.captures[1] == "linkage_name" || a.captures[1] == "name")
            keep = _attr_value(a.captures[2]) in wanted
        end
    end
    flush!()
    return units
end

function _build_unit!(units::Units, lines::Vector{String})
    stack = Die[]                 # stack[d + 1] is the open DIE at depth d
    cur = nothing
    for line in lines
        m = match(_DIE_LINE, line)
        if m !== nothing
            depth = parse(Int, m.captures[1])
            if m.captures[3] == "0"                   # end of a list of children
                cur = nothing
                continue
            end
            tag = m.captures[4] === nothing ? "" : String(m.captures[4])
            resize!(stack, min(length(stack), depth))
            parent = depth == 0 ? nothing : (length(stack) >= depth ? stack[depth] : nothing)
            die = Die(parse(Int, m.captures[2]; base = 16), tag, parent)
            parent === nothing ? (depth == 0 && push!(units.roots, die)) : push!(parent.children, die)
            length(stack) == depth && push!(stack, die)
            units.by_offset[die.offset] = die
            cur = die
            continue
        end
        cur === nothing && continue
        a = match(_ATTR_LINE, line)
        a === nothing && continue
        # The first spelling of an attribute stands: a DIE has each one once.
        get!(cur.attrs, String(a.captures[1]), _attr_value(a.captures[2]))
    end
    return units
end

# -----------------------------------------------------------------------------
# Reading attributes
# -----------------------------------------------------------------------------

"""
    _number(text) -> Union{Int,Nothing}

A constant as binutils prints it: decimal from one or two bytes and from LEB
forms, hex (`0x10000`) from four or eight bytes, with a parenthesised gloss
after it for the forms that have one (`1\t(public)`).
"""
function _number(text::AbstractString)::Union{Int,Nothing}
    v = _wide_number(text)
    return v isa Int ? v : nothing
end

"""
    _wide_number(text) -> Union{Int,String,Nothing}

`_number`, for a value that may not fit `Int64`: a niche discriminant is often
`0x8000000000000000`. Such a value comes back as its decimal string, which is
how the metadata carries it (the two JSON majors `Project.toml` allows read a
large integer differently, and one of them wraps it without an error).

Only a constant is a number. `2 byte block: 23 8 (DW_OP_plus_uconst: 8)` is a
location expression that starts with a digit, and is `nothing`.
"""
function _wide_number(text::AbstractString)::Union{Int,String,Nothing}
    m = match(r"^(-?)(0x[0-9a-fA-F]+|\d+)(?:\s*\([^()]*\))?$", strip(text))
    m === nothing && return nothing
    v = tryparse(Int128, m.captures[2])
    v === nothing && (u = tryparse(UInt128, m.captures[2]); u === nothing && return nothing; return string(u))
    m.captures[1] == "-" && (v = -v)
    return typemin(Int) <= v <= typemax(Int) ? Int(v) : string(v)
end

"A reference attribute's target offset (`<0x690>`), or `nothing`."
function _ref(text::AbstractString)::Union{Int,Nothing}
    m = match(r"^<0x([0-9a-fA-F]+)>", strip(text))
    return m === nothing ? nothing : parse(Int, m.captures[1]; base = 16)
end

_target(u::Units, die::Die, key::String) =
    haskey(die.attrs, key) ? (r = _ref(die.attrs[key]); r === nothing ? nothing : get(u.by_offset, r, nothing)) : nothing

"""
    _attr(units, die, key) -> Union{String,Nothing}

`die`'s attribute, or the one of the DIE it stands for: a definition takes its
name, linkage name and type from the declaration `DW_AT_specification` names,
and a concrete out-of-line instance from its `DW_AT_abstract_origin`.
"""
function _attr(u::Units, die::Die, key::String, depth::Int = 0)::Union{String,Nothing}
    haskey(die.attrs, key) && return die.attrs[key]
    depth > 8 && return nothing
    for link in ("abstract_origin", "specification")
        t = _target(u, die, link)
        t === nothing && continue
        v = _attr(u, t, key, depth + 1)
        v === nothing || return v
    end
    return nothing
end

"The DIE that declares what `die` defines: the end of its specification/abstract-origin chain."
function _declaration(u::Units, die::Die)::Die
    for _ in 1:8
        t = _target(u, die, "specification")
        t === nothing && (t = _target(u, die, "abstract_origin"))
        t === nothing && return die
        die = t
    end
    return die
end

const _SCOPE_TAGS = ("namespace", "structure_type", "union_type", "enumeration_type", "class_type")

"The names of the scopes around `die`, outermost first: namespaces and the types it is nested in."
function _scope(die::Die)::Vector{String}
    out = String[]
    p = die.parent
    while p !== nothing
        if p.tag in _SCOPE_TAGS
            pushfirst!(out, get(p.attrs, "name", "{unnamed}"))
        elseif p.tag == "subprogram"
            # A type or function local to a function: scoped by that function.
            pushfirst!(out, get(p.attrs, "name", "{fn}"))
        end
        p = p.parent
    end
    return out
end

_params(die::Die) = Die[c for c in die.children if c.tag == "formal_parameter"]
_is_definition(die::Die) = (haskey(die.attrs, "low_pc") || haskey(die.attrs, "ranges")) &&
                           !haskey(die.attrs, "declaration")

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

const _ENCODINGS = Dict(1 => "address", 2 => "boolean", 4 => "float", 5 => "signed",
                        6 => "signed_char", 7 => "unsigned", 8 => "unsigned_char", 16 => "utf")

"""
    _pointer_kind(name) -> String

What a pointer DIE's Rust spelling says it is: `ref` (`&T`), `mut_ref`
(`&mut T`), `const_ptr`, `mut_ptr`, `box` (`alloc::boxed::Box<T, A>`, which
owns its target), `fn` (a function pointer) or `other`.
"""
function _pointer_kind(name::AbstractString)::String
    startswith(name, "&mut ") && return "mut_ref"
    startswith(name, "&") && return "ref"
    startswith(name, "*const ") && return "const_ptr"
    startswith(name, "*mut ") && return "mut_ptr"
    startswith(name, "alloc::boxed::Box<") && return "box"
    (startswith(name, "fn(") || startswith(name, "unsafe fn(") ||
     startswith(name, "extern ") || startswith(name, "unsafe extern ")) && return "fn"
    return "other"
end

"""
    _fat_kind(fields) -> String

A fat pointer is a record of two words that rustc names like the pointer it
is: `slice` (`data_ptr`, `length`: `&[T]`, `&str`, `Box<[T]>`) or `dyn`
(`pointer`, `vtable`: `&dyn Trait`, `Box<dyn Trait>`). `""` for any other record.
"""
function _fat_kind(fields::Vector{Dict{String,Any}})::String
    length(fields) == 2 || return ""
    names = (fields[1]["name"], fields[2]["name"])
    names == ("data_ptr", "length") && return "slice"
    names == ("pointer", "vtable") && return "dyn"
    return ""
end

mutable struct _Table
    units::Units
    types::Dict{String,Any}                # key => description
    key_of::Dict{Int,String}               # DIE offset => key
    sig_of::Dict{String,String}            # key => what a second definition must equal
end

"""
    _type_key(table, die) -> String

Describe the type at `die` (and what it reaches) into the table and return
its key: the Rust spelling for a base type or pointer, the scoped name for a
record (`probe::Shape`, and `probe::Shape::Circle` for the record of one
variant, which rustc nests in the enum), `[T; N]` for an array.
"""
function _type_key(t::_Table, die::Die)::String
    haskey(t.key_of, die.offset) && return t.key_of[die.offset]
    u = t.units
    tag = die.tag
    name = get(die.attrs, "name", "")
    size = haskey(die.attrs, "byte_size") ? _number(die.attrs["byte_size"]) : nothing
    align = haskey(die.attrs, "alignment") ? _number(die.attrs["alignment"]) : nothing
    target() = (d = _target(u, die, "type"); d === nothing ? nothing : _type_key(t, d))

    if tag == "base_type"
        enc = haskey(die.attrs, "encoding") ? _number(die.attrs["encoding"]) : nothing
        return _enter!(t, die, name, Dict{String,Any}(
            "kind" => "base", "name" => name, "size" => size,
            "encoding" => get(_ENCODINGS, enc, "other")))
    elseif tag == "pointer_type" || tag == "reference_type"
        key = isempty(name) ? "{pointer@$(string(die.offset; base = 16))}" : name
        # Entered before its target is described: a record may point at itself.
        t.key_of[die.offset] = key
        tk = target()
        isempty(name) && (key = "*" * something(tk, "()"))
        t.key_of[die.offset] = key
        return _enter!(t, die, key, Dict{String,Any}(
            "kind" => "pointer", "name" => key, "pointer" => _pointer_kind(name),
            "target" => tk, "size" => something(size, 8)))
    elseif tag == "array_type"
        elem = target()
        count = nothing
        for c in die.children
            c.tag == "subrange_type" || continue
            if haskey(c.attrs, "count")
                count = _number(c.attrs["count"])
            elseif haskey(c.attrs, "upper_bound")
                ub = _number(c.attrs["upper_bound"])
                lb = haskey(c.attrs, "lower_bound") ? something(_number(c.attrs["lower_bound"]), 0) : 0
                ub === nothing || (count = ub - lb + 1)
            end
            break
        end
        key = "[" * something(elem, "?") * "; " * (count === nothing ? "?" : string(count)) * "]"
        # An array DIE carries no size of its own: it is its elements.
        if size === nothing && count !== nothing && elem !== nothing
            esize = get(get(t.types, elem, Dict{String,Any}()), "size", nothing)
            esize isa Integer && (size = count * esize)
        end
        return _enter!(t, die, key, Dict{String,Any}(
            "kind" => "array", "name" => key, "element" => elem, "count" => count,
            "size" => size))
    elseif tag == "subroutine_type"
        params = String[something(_type_key_of(t, c), "?") for c in die.children if c.tag == "formal_parameter"]
        ret = target()
        key = isempty(name) ? "fn(" * join(params, ", ") * ")" * (ret === nothing ? "" : " -> " * ret) : name
        return _enter!(t, die, key, Dict{String,Any}(
            "kind" => "subroutine", "name" => key, "parameters" => params, "return" => ret))
    elseif tag == "enumeration_type"
        key = join(vcat(_scope(die), isempty(name) ? "{enum}" : name), "::")
        t.key_of[die.offset] = key
        enumerators = Any[]
        for c in die.children
            c.tag == "enumerator" || continue
            push!(enumerators, Any[get(c.attrs, "name", ""),
                                   haskey(c.attrs, "const_value") ? _wide_number(c.attrs["const_value"]) : nothing])
        end
        return _enter!(t, die, key, Dict{String,Any}(
            "kind" => "c_enum", "name" => name, "scope" => _scope(die), "size" => size,
            "align" => align, "underlying" => target(), "enumerators" => enumerators))
    elseif tag in ("structure_type", "union_type", "class_type")
        key = join(vcat(_scope(die), isempty(name) ? "{record}" : name), "::")
        t.key_of[die.offset] = key
        if haskey(die.attrs, "declaration") && size === nothing
            return _enter!(t, die, key, Dict{String,Any}(
                "kind" => "opaque", "name" => name, "scope" => _scope(die)))
        end
        fields = Dict{String,Any}[]
        for c in die.children
            c.tag == "member" || continue
            push!(fields, _member(t, c))
        end
        desc = Dict{String,Any}("name" => name, "scope" => _scope(die), "size" => size,
                                "align" => align, "fields" => fields)
        vp = findfirst(c -> c.tag == "variant_part", die.children)
        if vp !== nothing
            desc["kind"] = "enum"
            _variant_part!(t, desc, die.children[vp])
        else
            desc["kind"] = tag == "union_type" ? "union" : "struct"
            fat = _fat_kind(fields)
            isempty(fat) || (desc["fat"] = fat)
            (startswith(name, "(") && endswith(name, ")")) && (desc["tuple"] = true)
        end
        return _enter!(t, die, key, desc)
    end
    # A typedef or a qualifier would be transparent; rustc emits neither. Any
    # other tag is a type this pass does not read, and is said to be one.
    key = "{DW_TAG_" * tag * (isempty(name) ? "" : " " * name) * "}"
    return _enter!(t, die, key, Dict{String,Any}("kind" => "unknown", "name" => name, "tag" => tag))
end

_type_key_of(t::_Table, die::Die) = (d = _target(t.units, die, "type"); d === nothing ? nothing : _type_key(t, d))

function _member(t::_Table, die::Die)::Dict{String,Any}
    m = Dict{String,Any}(
        "name" => get(die.attrs, "name", ""),
        "offset" => haskey(die.attrs, "data_member_location") ? _number(die.attrs["data_member_location"]) : nothing,
        "type" => _type_key_of(t, die))
    haskey(die.attrs, "alignment") && (m["align"] = _number(die.attrs["alignment"]))
    return m
end

# An enum that carries data: the discriminant's place and type, and each
# variant's value and record. A variant with no `DW_AT_discr_value` is the
# default: every value no other variant has (a niche, where the discriminant
# is a field of that variant).
function _variant_part!(t::_Table, desc::Dict{String,Any}, vp::Die)
    discr = _target(t.units, vp, "discr")
    if discr !== nothing
        d = _member(t, discr)
        tk = d["type"]
        tk === nothing || (d["size"] = get(get(t.types, tk, Dict{String,Any}()), "size", nothing))
        delete!(d, "name")
        desc["discriminant"] = d
    end
    variants = Dict{String,Any}[]
    for v in vp.children
        v.tag == "variant" || continue
        entry = Dict{String,Any}("default" => !haskey(v.attrs, "discr_value"))
        haskey(v.attrs, "discr_value") && (entry["discr_value"] = _wide_number(v.attrs["discr_value"]))
        for m in v.children
            m.tag == "member" || continue
            mm = _member(t, m)
            entry["name"] = mm["name"]
            entry["type"] = mm["type"]
            entry["offset"] = mm["offset"]
            break
        end
        push!(variants, entry)
    end
    desc["variants"] = variants
    return desc
end

# What two definitions under one key must agree on. A type is defined once per
# compile unit that uses it, so the same key arrives again with other offsets.
_signature(desc::Dict{String,Any}) = sprint() do io
    for k in ("kind", "size", "align", "encoding", "pointer", "target", "element", "count")
        print(io, k, '=', get(desc, k, nothing), ';')
    end
    for f in get(desc, "fields", Dict{String,Any}[])
        print(io, f["name"], '@', f["offset"], ':', f["type"], ';')
    end
    haskey(desc, "discriminant") && print(io, "discr@", desc["discriminant"]["offset"], ':',
                                          desc["discriminant"]["type"], ';')
    for v in get(desc, "variants", Dict{String,Any}[])
        print(io, get(v, "name", ""), '=', get(v, "discr_value", "default"), ':', get(v, "type", nothing), ';')
    end
    for e in get(desc, "enumerators", Any[])
        print(io, e[1], '=', e[2], ';')
    end
end

function _enter!(t::_Table, die::Die, key::AbstractString, desc::Dict{String,Any})::String
    key = String(key)
    t.key_of[die.offset] = key
    sig = _signature(desc)
    if haskey(t.types, key)
        # Two different types under one name: neither is the layout of that
        # name, and a function reaching it is refused (RustABI reads the mark).
        t.sig_of[key] == sig || (t.types[key]["conflict"] = true)
    else
        t.types[key] = desc
        t.sig_of[key] = sig
    end
    return key
end

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

"""
    describe_functions(units, wanted) -> (functions, types)

For each symbol of `wanted` that a subprogram DIE names: its Rust name, the
scopes around its declaration (modules, and the type a method belongs to),
its parameters in order with their types, and its return type. `types`
describes every type those signatures reach, keyed as `_type_key` says.

`functions` is keyed by symbol. A symbol no DIE names is absent, which the
caller reports: nothing is inferred for it.

A function's facts come from its DEFINITION (the DIE with code), whose
parameters carry the names; its name, linkage name and scope come from the
declaration that definition names, wherever in the unit it is. The tree is
built before anything is resolved, so the order of the DIEs does not matter.
"""
function describe_functions(u::Units, wanted::AbstractSet{<:AbstractString})
    t = _Table(u, Dict{String,Any}(), Dict{Int,String}(), Dict{String,String}())
    chosen = Dict{String,Die}()
    for off in sort!(collect(keys(u.by_offset)))
        die = u.by_offset[off]
        die.tag == "subprogram" || continue
        sym = _attr(u, die, "linkage_name")
        # No linkage name at all: a `#[no_mangle]` function, bound by its name.
        sym === nothing && (sym = _attr(u, die, "name"))
        (sym === nothing || !(sym in wanted)) && continue
        prev = get(chosen, sym, nothing)
        # A definition wins over a declaration; among equals, the first offset.
        if prev === nothing || (_is_definition(die) && !_is_definition(prev))
            chosen[sym] = die
        end
    end

    functions = Dict{String,Any}()
    for sym in sort!(collect(keys(chosen)))
        die = chosen[sym]
        decl = _declaration(u, die)
        params = _params(die)
        decl_params = decl === die ? params : _params(decl)
        note = nothing
        if isempty(params) && !isempty(decl_params)
            params = decl_params
        elseif !isempty(decl_params) && length(decl_params) != length(params)
            note = "the definition has $(length(params)) parameters and its declaration $(length(decl_params))"
        end
        plist = Dict{String,Any}[]
        for (i, p) in enumerate(params)
            td = _target(u, p, "type")
            td === nothing && (o = _target(u, p, "abstract_origin"); o === nothing || (td = _target(u, o, "type")))
            pname = _attr(u, p, "name")
            push!(plist, Dict{String,Any}("name" => something(pname, "arg$i"),
                                          "type" => td === nothing ? nothing : _type_key(t, td)))
        end
        rd = _target(u, die, "type")
        rd === nothing && decl !== die && (rd = _target(u, decl, "type"))
        scope = _scope(decl)
        owner = decl.parent !== nothing && decl.parent.tag in ("structure_type", "union_type", "enumeration_type") ?
                    _type_key(t, decl.parent) : nothing
        f = Dict{String,Any}(
            "name" => something(_attr(u, die, "name"), ""),
            "scope" => scope,
            "owner" => owner,
            "parameters" => plist,
            "return" => rd === nothing ? nothing : _type_key(t, rd),
            "defined" => _is_definition(die))
        note === nothing || (f["dwarf_note"] = note)
        functions[sym] = f
    end
    return (functions, t.types)
end

end # module RustDWARF
