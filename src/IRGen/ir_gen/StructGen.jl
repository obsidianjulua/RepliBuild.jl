module StructGen

using ..TypeUtils

export generate_struct_definitions, get_struct_type_string, get_struct_definition_string, is_struct_packed, get_julia_offsets, get_llvm_equivalent_type_string, get_llvm_aligned_type_string

"""
    is_struct_packed(info::Any) -> Bool

Determine if a struct is packed (dwarf_size == sum(member_sizes)).
"""
function is_struct_packed(info::Any)
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end
    
    if dwarf_size == 0
        return false
    end

    kind = get(info, "kind", "")
    if kind == "union" || kind == "enum"
        return false
    end
    
    members = get(info, "members", [])
    sum_size = 0
    
    for m in members
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        sum_size += m_size
    end
    
    return sum_size == dwarf_size
end

"""
    get_julia_offsets(info::Any, is_packed::Bool=false) -> Vector{Int}

Calculate the byte offsets of struct members according to Julia/C alignment rules.
Returns a vector of start offsets for each member.
"""
function get_julia_offsets(info::Any, is_packed::Bool=false)
    members = get(info, "members", [])
    offsets = Int[]
    current_offset = 0
    
    for m in members
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        
        if !is_packed
            # Alignment heuristic: alignment = min(size, 8)
            # Cap at 8 bytes (64-bit) usually
            align = m_size > 8 ? 8 : m_size
            align = align == 0 ? 1 : align
            
            # Add padding
            padding = (align - (current_offset % align)) % align
            current_offset += padding
        end
        
        push!(offsets, current_offset)
        
        current_offset += m_size
    end
    
    return offsets
end

"""
    _lookup_struct(all_structs, s_name) -> struct info or nothing

Find a struct's metadata by member-type name, trying the raw name and the
`__enum__`-prefixed key (enum definitions are stored under the prefixed key).
"""
function _lookup_struct(all_structs, s_name::AbstractString)
    all_structs === nothing && return nothing
    haskey(all_structs, s_name) && return all_structs[s_name]
    haskey(all_structs, "__enum__$(s_name)") && return all_structs["__enum__$(s_name)"]
    return nothing
end

"""
    _resolve_struct_member_type(mlir_t, all_structs, seen; alias_form) -> String

Resolve a struct-typed member reference for use inside another struct's body.

A member whose target struct is emitted as `!jlcs.c_struct` (packed) must NOT
be referenced via its alias or bare name from inside an `!llvm.struct` body:
the JLCS→LLVM type converter treats `!llvm.struct` as already legal and never
rewrites types nested in its body, so the JLCS type survives lowering and
SIGSEGVs `translateModuleToLLVMIR` (PtrLikeTypeInterface::getMemorySpace on a
non-LLVM type). Such members are inlined as their byte-identical LLVM packed
literal instead.

Non-packed targets keep the caller's convention: `alias_form=true` yields the
`!Struct_<name>` alias (definition-string context), `alias_form=false` leaves
the bare named ref untouched (equivalent/aligned-string context).
"""
function _resolve_struct_member_type(mlir_t::String, all_structs, seen::Set{String};
                                     alias_form::Bool, fallback_size::Int=0)
    # An array whose ELEMENT is a named struct hides that struct from the
    # top-level match below, and a body-less identified struct is exactly as
    # illegal inside `!llvm.array<...>` as it is on its own — the module still
    # fails to parse. Reachable only since `map_cpp_type` learned to render
    # `T[N]` as an array instead of collapsing it to `!llvm.ptr`.
    #
    # The element is always resolved to a CONCRETE type, never an alias: an
    # alias is a textual reference to a definition that may never have been
    # emitted (STL-internal and blocklisted structs are skipped), and a
    # `!jlcs.c_struct` alias nested in an `!llvm` body is the converter crash
    # this function's docstring describes. A struct element that survives as a
    # bare ref becomes its own exact DWARF size in bytes, so the array keeps its
    # true extent — layout stays right, only field addressability is given up,
    # which array elements never had anyway.
    am = match(r"^!llvm\.array<(\d+) x (.+)>$", mlir_t)
    if am !== nothing
        n = _parse_num(am.captures[1])
        if n !== nothing
            elem = _resolve_struct_member_type(String(am.captures[2]), all_structs, seen;
                                               alias_form=false)
            bare = match(r"^!llvm\.struct<\"([^\"]+)\">$", elem)
            if bare !== nothing
                einfo = _lookup_struct(all_structs, String(bare.captures[1]))
                esz = einfo === nothing ? nothing : _parse_num(get(einfo, "byte_size", nothing))
                esz === nothing && return "!llvm.array<$(max(fallback_size, 1)) x i8>"
                elem = "!llvm.array<$(max(esz, 1)) x i8>"
            end
            return "!llvm.array<$n x $elem>"
        end
    end

    m = match(r"^!llvm\.struct<\"(.+)\">$", mlir_t)
    m === nothing && return mlir_t
    s_name = String(m.captures[1])
    info = _lookup_struct(all_structs, s_name)

    # Member names a struct we have no definition for. Emitting the bare
    # `!llvm.struct<"name">` is a PARSE error — MLIR allows a body-less
    # identified struct only inside a recursive definition — and the alias form
    # is no better, since an unknown struct never gets an alias emitted either.
    # Either way the whole module fails to parse, which disables Tier 2 for the
    # ENTIRE library, not just this type.
    #
    # llama.cpp reached it through `FILE*`: glibc's `_G_fpos_t` embeds
    # `__mbstate_t`, whose anonymous union `_mbstate_t_value` is referenced by
    # name and defined nowhere. One system type nobody asked for, and every
    # Tier-2 thunk in the library was dead.
    #
    # Degrade to a correctly-sized byte region, the same treatment the by-value
    # cycle case gets below. Guarded on `all_structs !== nothing`: a caller that
    # supplied no map at all has always taken the alias/bare path and must keep
    # its behaviour — "no map" and "name absent from the map" are different
    # facts that `_lookup_struct` reports identically.
    if info === nothing && all_structs !== nothing
        return "!llvm.array<$(max(fallback_size, 1)) x i8>"
    end

    if info !== nothing && is_struct_packed(info)
        return _packed_literal_string(s_name, info, all_structs, seen)
    end
    if alias_form
        safe_name = replace(s_name, r"[^a-zA-Z0-9_]" => "_")
        return "!Struct_$(safe_name)"
    end

    # `alias_form=false` means this string gets embedded in a literal struct body
    # or a `!jlcs.c_struct` field list. A body-less identified struct is illegal
    # in BOTH — MLIR permits it only as a recursive self-reference — so returning
    # the bare ref unchanged (as this function used to) emits a module that
    # cannot be parsed, taking all of Tier 2 with it. The docstring's
    # "equivalent/aligned-string context" was never a context where a bare ref
    # was valid; it just wasn't reached until non-degrading structs started
    # emitting real bodies.
    #
    # Degrade to the target's exact DWARF size, so the enclosing struct's layout
    # is unchanged and only field addressability through this member is lost.
    info === nothing && return "!llvm.array<$(max(fallback_size, 1)) x i8>"
    sz = _parse_num(get(info, "byte_size", nothing))
    sz === nothing && return "!llvm.array<$(max(fallback_size, 1)) x i8>"
    return "!llvm.array<$(max(sz, 1)) x i8>"
end

"""
    _packed_literal_string(name, info, all_structs, seen) -> String

LLVM literal packed struct for a `!jlcs.c_struct`-classified struct — the
byte-identical, pure-LLVM spelling used when the struct is nested by value
inside another struct's body. Members that are themselves packed structs are
inlined recursively.
"""
function _packed_literal_string(name::String, info::Any, all_structs, seen::Set{String})
    if name in seen
        # A by-value cycle only comes from malformed metadata; degrade to a
        # correctly-sized byte blob rather than recursing forever.
        sz = try
            s = get(info, "byte_size", "0")
            startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s)
        catch
            0
        end
        return "!llvm.array<$(max(sz, 1)) x i8>"
    end
    push!(seen, name)
    member_types = String[]
    for m in get(info, "members", [])
        t = get(m, "c_type", "void*")
        m_size = try
            s = get(m, "size", 0)
            s isa String ? parse(Int, s) : s
        catch
            0
        end
        resolved = _resolve_struct_member_type(map_cpp_type(t), all_structs, seen;
                                               alias_form=false, fallback_size=m_size)
        # INVARIANT: a packed literal body contains no bare named struct refs.
        # `_resolve_struct_member_type(alias_form=false)` deliberately passes a
        # non-packed KNOWN struct through untouched, which is right in the
        # equivalent/aligned-string contexts it was written for — the name is
        # defined there. Inside this literal it is a body-less identified struct,
        # which MLIR rejects outside a recursive definition, and the module fails
        # to parse. glibc's `__mbstate_t` hit it: its anonymous union
        # `_mbstate_t_value` is a known NON-packed struct, so neither the
        # unknown-struct degrade above nor the packed-inline branch applied, and
        # `!llvm.struct<"_mbstate_t_value">` landed in the body verbatim.
        # A union has no LLVM spelling anyway (unions are emitted as byte arrays,
        # see get_struct_definition_string), so the byte region is also the
        # correct representation, not merely a safe one.
        if match(r"^!llvm\.struct<\"[^\"]+\">$", resolved) !== nothing
            resolved = "!llvm.array<$(max(m_size, 1)) x i8>"
        end
        push!(member_types, resolved)
    end
    delete!(seen, name)
    return "!llvm.struct<packed ($(join(member_types, ", ")))>"
end

#===========================================================================
 LLVM layout of an emitted MLIR type string
 ===========================================================================

Mirrors `abiSize`/`abiAlign` in `src/mlir/impl/JLCSPasses.cpp` over exactly the
type vocabulary this module emits. `nothing` means "outside that vocabulary" —
never zero. Treating an unmeasurable member as zero-sized is precisely the
arithmetic that produced the oversized structs `_dwarf_padded_members` exists
to prevent, so the caller must degrade instead of guessing.
=#

const _SCALAR_LAYOUT = Dict{String,Tuple{Int,Int}}(
    "i1"  => (1, 1), "i8"  => (1, 1), "i16" => (2, 2), "i32" => (4, 4),
    "i64" => (8, 8), "f32" => (4, 4), "f64" => (8, 8), "!llvm.ptr" => (8, 8),
)

_align_to(x::Int, a::Int) = a <= 1 ? x : ((x + a - 1) ÷ a) * a

function _parse_num(x)
    x === nothing && return nothing
    x isa Integer && return Int(x)
    s = String(strip(string(x)))
    isempty(s) && return nothing
    return try
        startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s)
    catch
        nothing
    end
end

"""
    _split_type_list(body) -> Vector{String}

Split an MLIR struct body on top-level commas. Bracket- and quote-aware:
`i32, !llvm.struct<(i32, i32)>` is two elements, not three.
"""
function _split_type_list(s::AbstractString)
    parts = String[]
    depth = 0
    in_str = false
    start = firstindex(s)
    i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]
        if in_str
            c == '"' && (in_str = false)
        elseif c == '"'
            in_str = true
        elseif c == '<' || c == '('
            depth += 1
        elseif c == '>' || c == ')'
            depth -= 1
        elseif c == ',' && depth == 0
            push!(parts, String(strip(s[start:prevind(s, i)])))
            start = nextind(s, i)
        end
        i = nextind(s, i)
    end
    tail = String(strip(s[start:end]))
    isempty(tail) || push!(parts, tail)
    return parts
end

"""
    _mlir_layout(t, all_structs, depth=0) -> Union{Tuple{Int,Int}, Nothing}

`(size, align)` of an emitted MLIR type string under the x86-64 data layout,
or `nothing` when the string cannot be measured.
"""
function _mlir_layout(t::AbstractString, all_structs, depth::Int=0)
    depth > 16 && return nothing
    t = String(strip(t))
    haskey(_SCALAR_LAYOUT, t) && return _SCALAR_LAYOUT[t]

    m = match(r"^!llvm\.array<(\d+) x (.+)>$", t)
    if m !== nothing
        el = _mlir_layout(String(m.captures[2]), all_structs, depth + 1)
        el === nothing && return nothing
        return (parse(Int, m.captures[1]) * el[1], el[2])
    end

    # A bare named ref carries no body — its size is the DWARF one. This is what
    # `_resolve_struct_member_type(alias_form=false)` leaves for a known
    # non-packed struct member, which is why measurement runs on that form.
    m = match(r"^!llvm\.struct<\"([^\"]+)\">$", t)
    if m !== nothing
        info = _lookup_struct(all_structs, String(m.captures[1]))
        info === nothing && return nothing
        sz = _parse_num(get(info, "byte_size", nothing))
        (sz === nothing || sz <= 0) && return nothing
        return (sz, _dwarf_struct_align(info, all_structs, depth + 1))
    end

    m = match(r"^!llvm\.struct<(?:\"[^\"]*\",\s*)?(packed\s*)?\((.*)\)\s*>$", t)
    if m !== nothing
        packed = m.captures[1] !== nothing
        cur = 0
        al = 1
        for e in _split_type_list(String(m.captures[2]))
            l = _mlir_layout(e, all_structs, depth + 1)
            l === nothing && return nothing
            if !packed
                cur = _align_to(cur, l[2])
                al = max(al, l[2])
            end
            cur += l[1]
        end
        packed || (cur = _align_to(cur, al))
        return (cur, packed ? 1 : al)
    end

    return nothing   # opaque, !jlcs.c_struct, or anything unrecognized
end

"""
    _dwarf_struct_align(info, all_structs, depth) -> Int

Alignment of a struct as this module emits it, derived structurally from its
DWARF members (the same rule `abiAlign` applies to the lowered type).
"""
function _dwarf_struct_align(info, all_structs, depth::Int)
    depth > 16 && return 8
    kind = get(info, "kind", "")
    if kind == "enum"
        u = map_cpp_type(String(get(info, "underlying_type", "int")))
        return get(_SCALAR_LAYOUT, u, (4, 4))[2]
    end
    kind == "union" && return 1   # unions are emitted as !llvm.array<N x i8>
    a = 1
    for m in get(info, "members", [])
        l = _mlir_layout(map_cpp_type(String(get(m, "c_type", "void*"))), all_structs, depth + 1)
        l === nothing || (a = max(a, l[2]))
    end
    return a
end

# Structs already reported as unmodellable — one line each, then capped. A
# whole-library wrap reaches every system and STL type the headers dragged in
# (llama.cpp: 101 of 2864, nearly all libstdc++ internals and ggml quant blocks
# whose fixed-size array members `map_cpp_type` still renders as pointers), and
# a wall of warnings is how a real one gets missed. The full set stays
# inspectable as `StructGen._LAYOUT_WARNED`.
const _LAYOUT_WARNED = Set{String}()
const _LAYOUT_WARN_CAP = 10

"""
    _dwarf_padded_members(name, info, member_types, all_structs, dwarf_size)
        -> Union{Vector{Union{Int,String}}, Nothing}

A layout plan: member indices interleaved with explicit `!llvm.array<N x i8>`
padding, such that every member lands on the byte offset DWARF recorded for it
and the struct's LLVM size is exactly `dwarf_size`. Indices rather than type
strings so the caller can project the plan onto whichever spelling of the
member list it emits.

The old rule appended ONE trailing filler of `dwarf_size - sum(member sizes)`.
That double-counts: LLVM already inserts interior alignment padding, and DWARF
reports enum members with `size = 0`, so the filler paid a second time for gaps
the natural layout had paid for already. Every non-packed struct with interior
padding therefore came out LARGER than the C type it models — llama.cpp's
`llama_context_params` was 200 bytes against a native 160, `llama_model_params`
80 against 72.

That is not a cosmetic mismatch. A MEMORY-class struct return is stored
straight into the caller's `Ref{T}` by the `llvm.emit_c_interface` wrapper, and
`Ref{T}` is sized from the JULIA struct, i.e. the true `dwarf_size`. So every
Tier-2 call returning such a struct wrote past a live Julia object — 34 bytes
past for `llama_context_default_params`, measured. Every member offset held the
right value, which is why it presented as intermittent corruption (a garbage
read here, a SIGSEGV there) rather than as a marshalling bug.

Returns `nothing` when the members cannot be laid out consistently — overlaps
(bitfields, union members), a member whose emitted type will not sit at its
DWARF offset, an unmeasurable type, or members that overrun `byte_size`. The
caller then degrades the whole struct to a correctly-sized byte region: opaque,
but never the wrong size. `sum > dwarf_size` used to fall through with no
filler at all and silently ship the oversized body; it now degrades too.
"""
function _dwarf_padded_members(name::String, info::Any, member_types::Vector{String},
                               all_structs, dwarf_size::Int)
    members = get(info, "members", [])
    (dwarf_size > 0 && length(members) == length(member_types)) || return nothing

    why(msg) = begin
        if !(name in _LAYOUT_WARNED)
            push!(_LAYOUT_WARNED, name)
            n = length(_LAYOUT_WARNED)
            if n <= _LAYOUT_WARN_CAP
                @warn "Struct `$name` cannot be modelled field-by-field in MLIR ($msg); " *
                      "emitting a $(dwarf_size)-byte opaque region instead. By-value " *
                      "crossings stay correctly sized, but its fields are not addressable " *
                      "from Tier-2 IR."
            elseif n == _LAYOUT_WARN_CAP + 1
                @warn "More structs cannot be modelled field-by-field; further reports " *
                      "suppressed. All of them are emitted at their exact DWARF size, so " *
                      "the ABI is unaffected. Full set: `RepliBuild.JLCSIRGenerator." *
                      "StructGen._LAYOUT_WARNED`."
            end
        end
        nothing
    end

    out = Union{Int,String}[]
    cur = 0
    struct_align = 1
    for (i, m) in enumerate(members)
        off = _parse_num(get(m, "offset", nothing))
        off === nothing && return why("member `$(get(m, "name", "#$i"))` has no DWARF offset")
        lay = _mlir_layout(member_types[i], all_structs)
        lay === nothing && return why("member `$(get(m, "name", "#$i"))` has unmeasurable type $(member_types[i])")
        msize, malign = lay
        off < cur && return why("member `$(get(m, "name", "#$i"))` at $off overlaps the previous member ending at $cur")
        if off > cur
            push!(out, "!llvm.array<$(off - cur) x i8>")
            cur = off
        end
        # LLVM bumps each element to its own alignment. If that moves the member
        # off the offset DWARF recorded, this body cannot model the struct.
        _align_to(cur, malign) == off ||
            return why("member `$(get(m, "name", "#$i"))` needs align $malign but sits at offset $off")
        push!(out, i)
        struct_align = max(struct_align, malign)
        cur = off + msize
        cur <= dwarf_size ||
            return why("member `$(get(m, "name", "#$i"))` ends at $cur, past byte_size $dwarf_size")
    end

    cur < dwarf_size && push!(out, "!llvm.array<$(dwarf_size - cur) x i8>")
    # LLVM rounds the struct up to its own alignment; a C struct's size is always
    # a multiple of it, so this has to be a no-op.
    _align_to(dwarf_size, struct_align) == dwarf_size ||
        return why("byte_size $dwarf_size is not a multiple of the struct alignment $struct_align")
    return out
end

"""
    _sized_member_types(name, info, all_structs; alias_form) -> (emit, measure)

Emitted member types for a struct body, plus the alias-free spelling of the same
list used for measurement. The two differ only where a known non-packed struct
member becomes `!Struct_<name>` instead of `!llvm.struct<"name">` — the same
type either way, but only the latter is measurable, since an alias cannot be
reversed to its DWARF key once sanitized.
"""
function _sized_member_types(name::String, info::Any, all_structs; alias_form::Bool)
    emit = String[]
    measure = String[]
    for m in get(info, "members", [])
        t = get(m, "c_type", "void*")
        m_size = something(_parse_num(get(m, "size", 0)), 0)
        mapped = map_cpp_type(t)
        push!(emit, _resolve_struct_member_type(mapped, all_structs, Set{String}([name]);
                                                alias_form=alias_form, fallback_size=m_size))
        push!(measure, alias_form ?
              _resolve_struct_member_type(mapped, all_structs, Set{String}([name]);
                                          alias_form=false, fallback_size=m_size) :
              emit[end])
    end
    return (emit, measure)
end

"""
    _apply_dwarf_layout(name, info, emit, measure, all_structs, dwarf_size, is_packed)
        -> Union{Vector{String}, Nothing}

Shared tail of the three body builders: for a non-packed struct with a known
size, replace the raw member list with the offset-driven padded one, or signal
(via `nothing`) that the caller should emit a `dwarf_size`-byte opaque region.
Packed structs are returned untouched — `is_struct_packed` means
`sum(member sizes) == byte_size`, so their bodies are exact by construction and
they travel the `!jlcs.c_struct` marshalling path, not this one.
"""
function _apply_dwarf_layout(name::String, info::Any, emit::Vector{String},
                             measure::Vector{String}, all_structs,
                             dwarf_size::Int, is_packed::Bool)
    (is_packed || dwarf_size <= 0) && return emit
    plan = _dwarf_padded_members(name, info, measure, all_structs, dwarf_size)
    plan === nothing && return nothing
    return String[p isa Int ? emit[p] : p for p in plan]
end

"""
    get_struct_definition_string(name::String, info::Any, all_structs=nothing) -> String

Get the MLIR type definition string for a struct. Pass the full
`struct_definitions` map as `all_structs` so struct-typed members that are
themselves packed (`!jlcs.c_struct`) can be inlined as LLVM literals instead
of alias references (see `_resolve_struct_member_type`).
"""
function get_struct_definition_string(name::String, info::Any, all_structs=nothing)
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end

    kind = get(info, "kind", "")
    if kind == "union"
        return "!llvm.array<$(dwarf_size) x i8>"
    elseif kind == "enum"
        underlying = get(info, "underlying_type", "i32")
        mlir_t = map_cpp_type(underlying)
        if isempty(mlir_t) || startswith(mlir_t, "!llvm.struct")
            mlir_t = "i32"
        end
        return "!llvm.struct<\"$(name)\", ($(mlir_t))>"
    end
    
    is_packed = is_struct_packed(info)

    # Struct-typed members become !Struct_<name> aliases (bare named refs are
    # only valid in recursive struct contexts in MLIR), except packed targets,
    # which are inlined as LLVM literals — their alias is a !jlcs.c_struct and
    # must not be nested inside an !llvm.struct body.
    (emit, measure) = _sized_member_types(name, info, all_structs; alias_form=true)
    laid_out = _apply_dwarf_layout(name, info, emit, measure, all_structs, dwarf_size, is_packed)
    laid_out === nothing && return "!llvm.array<$(dwarf_size) x i8>"
    member_types = laid_out

    if is_packed
        # Emit !jlcs.c_struct for packed structs
        offsets = get_julia_offsets(info, true) # packed offsets
        
        # Format: !jlcs.c_struct<"Name", [types], [[offsets]], packed=true>
        types_str = join(member_types, ", ")
        
        offsets_typed = ["$(o) : i64" for o in offsets]
        offsets_str = "[$(join(offsets_typed, ", "))]"
        
        return "!jlcs.c_struct<\"$(name)\", [$(types_str)], [$(offsets_str)], packed = true>"
    else
        # Standard LLVM struct
        if isempty(member_types)
             return "!llvm.struct<\"$(name)\", opaque>"
        else
             return "!llvm.struct<\"$(name)\", ($(join(member_types, ", ")))>"
        end
    end
end

"""
    get_struct_type_string(name::String, info::Any) -> String

Get the MLIR type reference string (alias).
"""
function get_struct_type_string(name::String, info::Any)
    def_str = get_struct_definition_string(name, info)
    if endswith(def_str, "opaque>")
        return def_str
    end
    # Sanitize name for alias
    safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
    return "!Struct_$(safe_name)"
end

"""
    get_llvm_equivalent_type_string(name::String, info::Any, all_structs=nothing) -> String

Get the LLVM literal struct type string corresponding to the struct.
Used for constructing values of packed structs. Pass `all_structs` so packed
struct-typed members get inlined as LLVM literals (their bare named ref would
otherwise be an opaque/JLCS type — see `_resolve_struct_member_type`).
"""
function get_llvm_equivalent_type_string(name::String, info::Any, all_structs=nothing)
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end

    is_packed = is_struct_packed(info)
    packed_attr = is_packed ? "packed " : ""

    (emit, measure) = _sized_member_types(name, info, all_structs; alias_form=false)
    laid_out = _apply_dwarf_layout(name, info, emit, measure, all_structs, dwarf_size, is_packed)
    laid_out === nothing && return "!llvm.array<$(dwarf_size) x i8>"
    member_types = laid_out

    if isempty(member_types)
         return "!llvm.struct<\"$(name)\", opaque>" # Fallback
    else
         # Return a literal struct (no name)
         return "!llvm.struct<$(packed_attr)($(join(member_types, ", "))) >"
    end
end

"""
    get_llvm_aligned_type_string(name::String, info::Any, all_structs=nothing) -> String

Get an LLVM literal struct type string WITHOUT packed attribute.
Used for thunk return types so Julia can read the struct with natural alignment.
Pass `all_structs` so packed struct-typed members get inlined as LLVM literals.
"""
function get_llvm_aligned_type_string(name::String, info::Any, all_structs=nothing)
    dwarf_size_str = get(info, "byte_size", "0")
    dwarf_size = try
        startswith(dwarf_size_str, "0x") ? parse(Int, dwarf_size_str[3:end], base=16) : parse(Int, dwarf_size_str)
    catch
        0
    end

    is_packed = is_struct_packed(info)

    (emit, measure) = _sized_member_types(name, info, all_structs; alias_form=false)
    laid_out = _apply_dwarf_layout(name, info, emit, measure, all_structs, dwarf_size, is_packed)
    laid_out === nothing && return "!llvm.array<$(dwarf_size) x i8>"
    member_types = laid_out

    if isempty(member_types)
         return "!llvm.struct<\"$(name)\", opaque>"
    else
         return "!llvm.struct<($(join(member_types, ", "))) >"
    end
end

"""
    generate_struct_definitions(structs::Any) -> (String, String)

Generate LLVM/JLCS struct type aliases and registration functions.
Returns (aliases_ir, registrations_ir).
"""
function generate_struct_definitions(structs::Any)
    io_aliases = IOBuffer()
    io_regs = IOBuffer()
    
    println(io_aliases, "// Struct Aliases")
    println(io_regs, "// Struct Definitions (Registration)")
    
    nodes = String[]
    node_map = Dict{String, Any}() # Name -> Info
    
    for (name, info) in structs
        if name in ["int", "float", "double", "bool", "char", "void"]
            continue
        end

        effective_name = name
        if startswith(name, "__enum__")
            effective_name = replace(name, "__enum__" => "")
        end
        
        push!(nodes, effective_name)
        node_map[effective_name] = info
    end
    
    deps = Dict{String, Set{String}}()
    
    for name in nodes
        info = node_map[name]
        d = Set{String}()
        deps[name] = d
        
        kind = get(info, "kind", "")
        if kind == "enum" || kind == "union"
            continue
        end
        
        members = get(info, "members", [])
        for m in members
            t = get(m, "c_type", "void*")
            if endswith(t, "*") || contains(t, "*")
                continue
            end
            
            mlir_t = map_cpp_type(t)
            # Use triple-quoted regex to avoid escape issues
            m_match = match(r"""!llvm.struct<\"([^\"]+)\">""", mlir_t)
            if m_match !== nothing
                dep_name = m_match.captures[1]
                if haskey(node_map, dep_name) && dep_name != name
                    push!(d, dep_name)
                end
            end
        end
    end
    
    sorted_nodes = String[]
    visited = Set{String}()
    stack = Set{String}()
    
    function visit(n)
        if n in visited
            return
        end
        if n in stack
            return
        end
        
        push!(stack, n)
        if haskey(deps, n)
            for d in deps[n]
                visit(d)
            end
        end
        delete!(stack, n)
        
        push!(visited, n)
        push!(sorted_nodes, n)
    end

    for n in nodes
        visit(n)
    end
    
    # Collect all alias names that will be defined so we can validate references
    defined_aliases = Set{String}()
    for name in sorted_nodes
        info = node_map[name]
        def_str = get_struct_definition_string(name, info, node_map)
        if !endswith(def_str, "opaque>")
            safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
            push!(defined_aliases, "!Struct_$(safe_name)")
        end
    end

    for name in sorted_nodes
        info = node_map[name]
        def_str = get_struct_definition_string(name, info, node_map)
        
        if endswith(def_str, "opaque>")
            continue
        end
        
        # Replace any undefined alias references with !llvm.ptr
        def_str = replace(def_str, r"!Struct_[A-Za-z0-9_]+" => m -> m in defined_aliases ? m : "!llvm.ptr")

        # Emit alias
        safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")
        alias_name = "!Struct_$(safe_name)"
        println(io_aliases, "$(alias_name) = $(def_str)")

        # Use a dummy function to register usage (safe_name avoids <> in func name)
        println(io_regs, "func.func private @__def_$(safe_name)($(alias_name)) -> ()")
    end
    
    return (String(take!(io_aliases)), String(take!(io_regs)))
end
end