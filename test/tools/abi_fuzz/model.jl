# test/tools/abi_fuzz/model.jl — the shape model behind test/tools/abi_fuzz.jl.
#
# Pure Julia, no RepliBuild: the parent (which builds and wraps) and the child
# (which calls) both include this file, so the two agree on the unit, on the
# value every leaf gets, and on the order of the C table. It holds:
#
#   * the model types (records, enums, functions);
#   * `generate(seed)`, one random translation unit per seed (`Random.Xoshiro`);
#   * `leaves`, the scalar locations of a record as C names them (`.b.c[2]`);
#   * the value formulas, so a value the callee recorded or returned can be
#     recomputed from (seed, function, leaf) alone;
#   * `emit_main`/`emit_twin`, the C text. One text is compiled twice: as `.c`
#     (Tier 3, ccall) and as `.cpp` (C++ linkage, Tier 2 thunks). The
#     `__cplusplus` switches only touch the harness data, never the API.
#
# The C compiler is the oracle. Sizes, offsets and bitfield positions are never
# computed here: the unit exports them (`af_sizes`, `af_offs`, `af_bfp`) and the
# child reads them through `cglobal`. Size estimates in this file only bias the
# generator towards interesting sizes.

module AbiFuzzModel

using Random
using Printf

export Unit, Record, Member, EnumDef, Func, Param, CT, Prim, EnumT, RecT, ArrT, Leaf,
       PathEntry, Slot
export generate, leaves, paths, call_slots, bitfield_probes, named_records, planned_checks,
       emit_main, emit_twin, emit_enum, emit_record, src_names, unit_tag, rec_spell, type_spell, prim_signed,
       isptrparam, arg_value, ret_value, access_value, enum_index, recorded_bits,
       max_slots, PRIM_ORDER, PRIM_C, lit_width, Plan

# ── Scalar kinds ─────────────────────────────────────────────────────────────

# The order of `af_prims` in the exported table. Never reorder: the child reads
# the table by position.
const PRIM_ORDER = [:char, :schar, :uchar, :short, :ushort, :int, :uint, :long,
                    :ulong, :llong, :ullong, :float, :double, :bool, :ptr]

const PRIM_C = Dict(
    :char => "char", :schar => "signed char", :uchar => "unsigned char",
    :short => "short", :ushort => "unsigned short", :int => "int",
    :uint => "unsigned int", :long => "long", :ulong => "unsigned long",
    :llong => "long long", :ullong => "unsigned long long", :float => "float",
    :double => "double", :bool => "AF_BOOL", :ptr => "void*", :ldouble => "long double")

# `long double` is only ever a parameter or a return, never a record member. It
# is the scalar CLAUDE.md lists as an ABI trap on both tiers, so it is here to
# prove the expected-refusal path, not to be wrapped.

# `char`'s signedness is the platform's: the child reads it from the table
# (af_meta[7]) and never from here.
prim_signed(k::Symbol) = k in (:schar, :short, :int, :long, :llong)

# Natural sizes for the generator's size estimates. `long` is the one scalar
# whose width the platform decides (LLP64 vs LP64).
_est_size(k::Symbol) = k in (:char, :schar, :uchar, :bool) ? 1 :
                       k in (:short, :ushort) ? 2 :
                       k in (:int, :uint, :float) ? 4 :
                       k in (:long, :ulong) ? sizeof(Clong) :
                       k == :ldouble ? 16 : 8

"The width the generator writes a return literal at; matches the host compiler."
lit_width(t) = t isa Prim ? _est_size(t.k) : 8

# ── Model ────────────────────────────────────────────────────────────────────

abstract type CT end
struct Prim  <: CT; k::Symbol; end              # scalar (incl. void*)
struct EnumT <: CT; e::Int; end                 # unit.enums[e]
struct RecT  <: CT; r::Int; end                 # unit.records[r]
struct ArrT  <: CT; elem::CT; n::Int; end       # fixed array; n == 0: pointer param

mutable struct Member
    name::String     # "" = anonymous record member or unnamed bitfield
    ty::CT
    bits::Int        # -1 = not a bitfield; >= 0 = width (0 only when unnamed)
end

mutable struct Record
    name::String     # the name C code uses: typedef name, or the tag if tagonly
    tag::String      # struct/union tag; "" = untagged
    tagonly::Bool    # no typedef: spelled `struct tag`
    isunion::Bool
    members::Vector{Member}
    packed::Bool     # __attribute__((packed))
    pack::Int        # #pragma pack(n); 0 = none
    anon::Bool       # C11 anonymous member type, defined inline in its parent
    active::Int      # unions: the member that is set, recorded and returned
end

mutable struct EnumDef
    name::String     # typedef name
    tag::String      # "" = untagged
    names::Vector{String}
    values::Vector{Int64}
    fixed::Symbol    # :none, or the prim kind of a fixed underlying type
    twin::Int        # 0, or 1/2 for the identical-member-set pair (TU 1/2)
end

mutable struct Param
    name::String
    ty::CT           # Prim, EnumT, RecT (by value), or ArrT(RecT, 0) (pointer)
end

mutable struct Func
    name::String
    params::Vector{Param}
    ret::Union{Nothing,CT}
    id::Int
    tu::Int          # 1 = main unit, 2 = the twin unit
end

mutable struct Unit
    seed::Int
    ident::Bool      # identifier-stress names
    records::Vector{Record}
    enums::Vector{EnumDef}
    funcs::Vector{Func}
end

"""
Everything the child needs for one wrapped unit, serialized by the parent.
`checks` is what `planned_checks` promised; `only`/`skip` narrow a rerun.
"""
struct Plan
    unit::Unit
    lang::Symbol                     # :c or :cpp
    tag::String                      # "17/c": the prefix of every PROBE name
    libpath::String
    wrapper::String
    modname::String
    mangled::Dict{String,String}     # C function name => linker symbol
    neuter::String                   # "" or a child-side neuter (see neuter.jl)
    checks::Vector{String}
    only::Vector{String}
    skip::Vector{String}
end

# A pointer parameter is an `ArrT(RecT(r), 0)`: the model has no pointer type of
# its own, because a pointer member is always the opaque `void*` scalar.
isptrparam(t::CT) = t isa ArrT && t.n == 0

unit_tag(u::Unit) = string(u.seed, u.ident ? "n" : "")

named_records(u::Unit) = [i for (i, r) in enumerate(u.records) if !r.anon]

# ── Spelling ─────────────────────────────────────────────────────────────────

rec_spell(u::Unit, r::Int) = let rec = u.records[r]
    rec.tagonly ? string(rec.isunion ? "union " : "struct ", rec.tag) : rec.name
end

function type_spell(u::Unit, t::CT)
    t isa Prim  && return PRIM_C[t.k]
    t isa EnumT && return u.enums[t.e].name
    t isa RecT  && return rec_spell(u, t.r)
    error("type_spell: $t")
end

# ── Leaves: scalar locations as C names them ─────────────────────────────────

"""
One scalar location inside a record. `cpath` is the C access suffix (`.b.c[2]`,
anonymous members transparent as C11 makes them); `steps` the model route to it,
`(:m, record, member)` or `(:i, index, 0)`; `ty` the scalar type; `bits` the
bitfield width or -1; `signed` the integer interpretation (plain `char` says
false here and takes the platform's answer from the table).
"""
struct Leaf
    cpath::String
    steps::Vector{Tuple{Symbol,Int,Int}}
    ty::CT
    bits::Int
    signed::Bool
end

function _leaves!(out, u::Unit, r::Int, cprefix::String, steps)
    rec = u.records[r]
    for (i, m) in enumerate(rec.members)
        rec.isunion && i != rec.active && continue
        m.bits == 0 && continue
        if isempty(m.name)
            m.bits > 0 && continue                       # unnamed bitfield: padding
            _leaves!(out, u, (m.ty::RecT).r, cprefix, vcat(steps, [(:m, r, i)]))
            continue
        end
        _visit!(out, u, m.ty, m.bits, cprefix * "." * m.name, vcat(steps, [(:m, r, i)]))
    end
    return out
end

function _visit!(out, u::Unit, t::CT, bits::Int, cpath::String, steps)
    if t isa Prim
        push!(out, Leaf(cpath, steps, t, bits, prim_signed(t.k)))
    elseif t isa EnumT
        push!(out, Leaf(cpath, steps, t, bits, true))
    elseif t isa RecT
        _leaves!(out, u, t.r, cpath, steps)
    elseif t isa ArrT
        for j in 0:t.n-1
            _visit!(out, u, t.elem, -1, cpath * "[$j]", vcat(steps, [(:i, j, 0)]))
        end
    end
    return out
end

leaves(u::Unit, r::Int) = _leaves!(Leaf[], u, r, "", Tuple{Symbol,Int,Int}[])

"""
A C-nameable member path of a named record whose offset the table carries:
each direct member (an anonymous member's names injected, as C does) and every
non-bitfield leaf. `direct` marks the ones `member_offset` should answer.
"""
struct PathEntry
    cpath::String
    direct::Bool
end

function _direct!(out, u::Unit, r::Int)
    for m in u.records[r].members
        m.bits >= 0 && continue                          # bitfields have no offsetof
        if isempty(m.name)
            _direct!(out, u, (m.ty::RecT).r)             # injected names
        else
            push!(out, "." * m.name)
        end
    end
    return out
end

function paths(u::Unit, r::Int)
    out = PathEntry[]
    seen = Set{String}()
    for p in _direct!(String[], u, r)
        p in seen || (push!(seen, p); push!(out, PathEntry(p, true)))
    end
    for l in leaves(u, r)
        l.bits >= 0 && continue
        l.cpath in seen || (push!(seen, l.cpath); push!(out, PathEntry(l.cpath, false)))
    end
    return out
end

"Every named bitfield leaf of every named record, in table (`af_bfp`) order."
function bitfield_probes(u::Unit)
    out = Tuple{Int,Leaf}[]
    for r in named_records(u), l in leaves(u, r)
        l.bits > 0 && push!(out, (r, l))
    end
    return out
end

# ── Values ───────────────────────────────────────────────────────────────────
#
# Every value is a pure function of (seed, salt, id, ordinal): salt 1 is an
# argument, 2 a return, 3 the no-call accessor checks. The ordinal's low byte is
# folded into integers, so two leaves of one call never share a value while the
# width allows it, and no integer leaf is 0.

function _mix(x::UInt64)
    x += 0x9e3779b97f4a7c15
    z = x
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end
_h(seed, salt, id, ord) = _mix(_mix(_mix(UInt64(seed) ⊻ (UInt64(salt) << 56)) ⊻ UInt64(id)) ⊻ UInt64(ord))

"""
    leaf_value(u, ty, bits, signed, width, h, ord) -> value

The value of one scalar: an `Int128` for integers (already wrapped to the C
width and sign; plain `char` is generated unsigned), `Float32`/`Float64`, a
`UInt64` for a pointer, an enumerator's `Int64` for an enum. `width` is the C
byte width, which the child reads from the table.
"""
function leaf_value(u::Unit, ty::CT, bits::Int, signed::Bool, width::Int, h::UInt64, ord::Int)
    if ty isa EnumT
        e = u.enums[ty.e]
        return e.values[1 + Int((h + UInt64(ord)) % UInt64(length(e.values)))]
    end
    k = (ty::Prim).k
    if k == :float
        f = Float32(ord + 1) + Float32((h >> 8) % 1024) / 1024f0
        return isodd(h) ? -f : f
    elseif k == :double || k == :ldouble
        d = Float64(ord + 1) + Float64((h >> 8) % (1 << 20)) / Float64(1 << 20)
        return isodd(h) ? -d : d
    elseif k == :ptr
        return UInt64(0x1000) * UInt64(ord + 1) + ((h >> 20) % UInt64(0x1000)) * UInt64(0x10000)
    elseif k == :bool
        return Int128(isodd(ord) ⊻ isodd(h >> 3) ? 0 : 1)
    end
    if bits > 0
        span = Int128(1) << bits
        v = (Int128(ord + 1) + Int128((h >> 11) % UInt64(1 << 20))) % span
        signed && v >= span >> 1 && (v -= span)
        v == 0 && (v = signed ? Int128(-1) : Int128(1))
        return v
    end
    nb = 8 * width
    raw = ((Int128(h) & ~Int128(0xFF)) | Int128((ord + 1) & 0xFF)) & ((Int128(1) << nb) - 1)
    raw == 0 && (raw = 1)
    signed && raw >= (Int128(1) << (nb - 1)) && (raw -= Int128(1) << nb)
    return raw
end

arg_value(u::Unit, f::Func, ord::Int, ty, bits, signed, width) =
    leaf_value(u, ty, bits, signed, width, _h(u.seed, 1, f.id, ord), ord)
ret_value(u::Unit, f::Func, ord::Int, ty, bits, signed, width) =
    leaf_value(u, ty, bits, signed, width, _h(u.seed, 2, f.id, ord), ord)
access_value(u::Unit, r::Int, ord::Int, ty, bits, signed, width) =
    leaf_value(u, ty, bits, signed, width, _h(u.seed, 3, 1000 + r, ord), ord)

enum_index(u::Unit, e::Int, v) = findfirst(==(Int64(v)), u.enums[e].values)

"""
    recorded_bits(v, nbits, signed) -> UInt64

What `AF_RS`/`AF_RU` store for an integer `v` held in `nbits` bits: C widens it
by the type's own signedness. A bitfield passes its width, anything else 8×size.
"""
function recorded_bits(v::Integer, nbits::Int, signed::Bool)
    m = (UInt128(1) << nbits) - 1
    x = UInt128(mod(Int128(v), Int128(1) << nbits))
    signed && nbits < 128 && (x >> (nbits - 1)) & 1 == 1 && (x |= ~m)
    return UInt64(x & 0xFFFFFFFFFFFFFFFF)
end

# ── Call slots: what each callee records, in order ──────────────────────────

"Argument `param`, `leaf` within it (`nothing` for a scalar), ordinal `ord`."
struct Slot
    param::Int
    leaf::Union{Nothing,Leaf}
    ord::Int
end

function call_slots(u::Unit, f::Func)
    out = Slot[]
    for (pi, p) in enumerate(f.params)
        t = p.ty
        if t isa RecT || isptrparam(t)
            r = t isa RecT ? t.r : ((t::ArrT).elem::RecT).r
            for l in leaves(u, r)
                push!(out, Slot(pi, l, length(out)))
            end
        else
            push!(out, Slot(pi, nothing, length(out)))
        end
    end
    return out
end

max_slots(u::Unit) = maximum((length(call_slots(u, f)) for f in u.funcs); init = 0)

# ── Names ────────────────────────────────────────────────────────────────────

# Identifier-stress pools. Every entry is a valid identifier in C and in C++17
# (C++ keywords such as `using`, `mutable`, `try`, `catch`, `export` are left out, so one text
# compiles both ways). They aim at Julia: keywords, Base/Core bindings, names
# the generated wrapper itself defines, all-underscore and leading-underscore
# names, and spellings that differ only in case.
const IDENT_JULIA_KEYWORDS = ["end", "function", "begin", "quote", "let", "local",
    "global", "macro", "abstract", "primitive", "baremodule",
    "elseif", "finally", "where", "in", "isa", "outer", "type", "module", "import"]
const IDENT_BASE = ["Int", "Ptr", "Ref", "Any", "String", "Nothing", "Vector",
    "Tuple", "NTuple", "Symbol", "Type", "Function", "Union", "Base", "Core",
    "Main", "error", "get", "string", "length", "print", "show", "convert",
    "unsafe_load", "pointer", "nothing", "missing", "Inf", "NaN", "pi", "im",
    "eval", "include", "ccall", "cglobal", "getfield", "fieldnames", "Cint",
    "Cvoid", "Cstring", "C_NULL", "UInt8", "Float64", "Bool", "Char", "Libdl",
    "GC", "isbits", "ntuple", "zero", "one", "typemax", "Array", "Dict"]
const IDENT_WRAPPER = ["with", "setproperty", "setproperties", "getproperty",
    "struct_size", "member_offset", "dispatch_tier", "libc_free", "LIBRARY_PATH",
    "LIB_HANDLE", "METADATA", "STRUCT_SIZES", "STRUCT_OFFSETS", "DISPATCH_TIER",
    "BUILD_TARGET", "LIBRARY_SHA", "RepliBuild", "unsafe_convert", "__init__",
    "_check_build_identity", "_flush_cstdout", "LTO_IR", "THUNKS_LIBRARY_PATH"]
const IDENT_UNDERSCORE = ["_", "__", "___", "_x", "__x", "_X", "_data", "data",
    "_pad_0", "_pad_tail", "_anon1", "_anon2", "_1", "x_", "_end"]
const IDENT_CASE = [("point", "Point", "POINT"), ("vec", "Vec", "VEC"),
                    ("node", "Node", "NODE"), ("cell", "Cell", "CELL")]
const IDENT_POOL = vcat(IDENT_JULIA_KEYWORDS, IDENT_BASE, IDENT_WRAPPER, IDENT_UNDERSCORE,
                        [c for t in IDENT_CASE for c in t])

# Never generated: C/C++ keywords, the harness's own names, and what the
# preamble's headers define.
const _RESERVED = Set(["af_rec", "af_rec_n", "af_rec_fn", "af_meta", "af_prims",
    "af_esizes", "af_sizes", "af_offs", "af_bfp", "af_ready", "af_all", "af_r",
    "af_init_bfp", "af_bfp_ctor_", "af_bfp_init_", "af_c", "s", "main", "memset",
    "memcpy", "offsetof", "size_t", "uintptr_t", "intptr_t", "ptrdiff_t", "NULL",
    "AF_BOOL", "AF_TW0", "AF_TW1", "AF_TW2", "TwinA", "TwinB", "af_twin_a", "af_twin_b",
    "int", "char", "float", "double", "long", "short", "signed", "unsigned", "void",
    "struct", "union", "enum", "typedef", "const", "static", "extern", "return",
    "if", "else", "for", "while", "do", "switch", "case", "default", "break",
    "continue", "goto", "sizeof", "volatile", "register", "auto", "inline",
    "restrict", "_Bool", "bool", "true", "false", "typeof", "asm", "class", "new",
    "delete", "this", "template", "operator", "namespace", "private", "public",
    "protected", "virtual", "friend", "throw", "try", "catch", "export", "explicit",
    "nullptr", "and", "or", "not", "xor", "bitand", "bitor", "compl", "and_eq",
    "or_eq", "xor_eq", "not_eq", "wchar_t", "char16_t", "char32_t", "char8_t",
    "alignas", "alignof", "decltype", "noexcept", "static_assert", "thread_local",
    "concept", "requires", "consteval", "constinit", "constexpr", "typeid",
    "typename", "dynamic_cast", "static_cast", "reinterpret_cast", "const_cast",
    "mutable_"])

"""
Identifier allocation. In C, typedefs, functions and enumerators share one
namespace and members live in their record's; in C++ a tag is also a type name,
and a member that reuses a type name it was declared with "changes its meaning".
So in identifier mode every name avoids every ordinary name (`ordinary`), and
ordinary names avoid every member or parameter name used so far (`members`).
"""
mutable struct Names
    ordinary::Set{String}
    members::Set{String}
    ident::Bool
end

function fresh!(n::Names, rng, base::String)
    if n.ident && rand(rng) < 0.85
        for _ in 1:10
            c = IDENT_POOL[rand(rng, 1:length(IDENT_POOL))]
            (c in n.ordinary || c in n.members || c in _RESERVED) && continue
            push!(n.ordinary, c)
            return c
        end
    end
    i = 0
    while true
        c = i == 0 ? base : string(base, "_", i)
        if !(c in n.ordinary) && !(c in n.members) && !(c in _RESERVED)
            push!(n.ordinary, c)
            return c
        end
        i += 1
    end
end

# A member or parameter name: unique within `localset`, never an ordinary name.
function member_name!(n::Names, localset::Set{String}, rng, base::String)
    if n.ident && rand(rng) < 0.8
        for _ in 1:10
            c = IDENT_POOL[rand(rng, 1:length(IDENT_POOL))]
            (c in localset || c in n.ordinary || c in _RESERVED) && continue
            push!(localset, c); push!(n.members, c)
            return c
        end
    end
    c = base
    while c in localset || c in n.ordinary || c in _RESERVED
        c *= "_"
    end
    push!(localset, c); push!(n.members, c)
    return c
end

# ── Generation ───────────────────────────────────────────────────────────────

const _SIZE_TARGETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18,
                       20, 23, 24, 28, 31, 32, 33, 40, 48]
const _SIZE_WEIGHTS = [3, 4, 3, 5, 3, 3, 3, 6, 3, 3, 2, 5, 2, 3, 6, 4, 3, 3, 2, 4,
                       2, 2, 4, 2, 2, 1]

function _pick_weighted(rng, items, weights)
    t = rand(rng) * sum(weights)
    acc = 0.0
    for (it, w) in zip(items, weights)
        acc += w
        t <= acc && return it
    end
    return items[end]
end

# Rough natural-layout size, only to steer member generation towards a target.
function _est(u::Unit, t::CT)::Tuple{Int,Int}
    t isa Prim && return (_est_size(t.k), _est_size(t.k))
    if t isa EnumT
        s = u.enums[t.e].fixed === :none ? 4 : _est_size(u.enums[t.e].fixed)
        return (s, s)
    end
    t isa ArrT && return let (s, a) = _est(u, t.elem); (s * max(t.n, 1), a) end
    t isa RecT && return _est_rec(u, t.r)
    return (8, 8)
end
function _est_rec(u::Unit, r::Int)
    rec = u.records[r]
    sz = 0; al = 1
    for m in rec.members
        (s, a) = m.bits >= 0 ? (cld(max(m.bits, 1), 8), 1) : _est(u, m.ty)
        a = rec.packed ? 1 : (rec.pack > 0 ? min(a, rec.pack) : a)
        al = max(al, a)
        sz = rec.isunion ? max(sz, s) : (cld(sz, a) * a + s)
    end
    return (cld(max(sz, 1), al) * al, al)
end

const _PRIM_W = Dict(:char => 3, :schar => 2, :uchar => 3, :short => 3, :ushort => 2,
                     :int => 5, :uint => 3, :long => 1, :ulong => 1, :llong => 3,
                     :ullong => 2, :float => 4, :double => 4, :bool => 2, :ptr => 3)

function _rand_prim(rng; room::Int = 64)
    ks = [k for k in PRIM_ORDER if _est_size(k) <= max(room, 1)]
    isempty(ks) && (ks = [:char])
    return _pick_weighted(rng, ks, [_PRIM_W[k] for k in ks])
end

const _BF_BASES = [:int, :uint, :uchar, :ushort, :short, :llong, :ullong, :bool, :char]
const _BF_W     = [5, 5, 2, 2, 2, 2, 2, 1, 1]

function _gen_bitfield(rng, n::Names, localset, i)
    base = _pick_weighted(rng, _BF_BASES, _BF_W)
    maxw = base == :bool ? 1 : 8 * _est_size(base)
    if rand(rng) < 0.12
        # unnamed: a zero width closes the storage unit, a nonzero width is
        # padding the callee never sees
        return Member("", Prim(base == :bool ? :uint : base),
                      rand(rng) < 0.5 ? 0 : rand(rng, 1:min(maxw, 7)))
    end
    w = rand(rng) < 0.6 ? rand(rng, 1:min(maxw, 8)) : rand(rng, 1:maxw)
    return Member(member_name!(n, localset, rng, "m$i"), Prim(base), w)
end

function _gen_anon!(u::Unit, rng, n::Names, parent_local::Set{String})
    isunion = rand(rng) < 0.45
    ms = Member[]
    # C11 injects these names into the parent, so they share its namespace
    for i in 1:rand(rng, 1:3)
        if rand(rng) < 0.2
            push!(ms, _gen_bitfield(rng, n, parent_local, 100 + i))
        else
            push!(ms, Member(member_name!(n, parent_local, rng, "a$(length(u.records))_$i"),
                             Prim(_rand_prim(rng; room = 8)), -1))
        end
    end
    any(m -> !isempty(m.name), ms) ||
        push!(ms, Member(member_name!(n, parent_local, rng, "a$(length(u.records))_9"), Prim(:int), -1))
    named = [i for (i, m) in enumerate(ms) if !isempty(m.name)]
    push!(u.records, Record("", "", false, isunion, ms, false, 0, true,
                            isunion ? named[rand(rng, 1:length(named))] : 0))
    return length(u.records)
end

function _gen_record!(u::Unit, rng, n::Names)
    prior = named_records(u)
    isunion = rand(rng) < 0.18
    target = _pick_weighted(rng, _SIZE_TARGETS, _SIZE_WEIGHTS)
    lay = rand(rng)
    packed = lay < 0.12
    pack = (!packed && lay < 0.26) ? rand(rng, [1, 2, 4]) : 0
    rec = Record("", "", false, isunion, Member[], packed, pack, false, 0)
    push!(u.records, rec)
    me = length(u.records)
    localset = Set{String}()
    i = 0
    while length(rec.members) < 9
        i += 1
        (est, _) = _est_rec(u, me)
        if !isempty(rec.members)
            est >= target && break
            isunion && length(rec.members) >= 4 && break
        end
        room = max(target - est, 1)
        x = rand(rng)
        if x < 0.13
            push!(rec.members, _gen_bitfield(rng, n, localset, i))
        elseif x < 0.20 && any(e -> e.twin == 0, u.enums)
            es = [k for (k, e) in enumerate(u.enums) if e.twin == 0]
            push!(rec.members, Member(member_name!(n, localset, rng, "m$i"), EnumT(es[rand(rng, 1:length(es))]), -1))
        elseif x < 0.31 && !isempty(prior)
            push!(rec.members, Member(member_name!(n, localset, rng, "m$i"), RecT(prior[rand(rng, 1:length(prior))]), -1))
        elseif x < 0.42
            k = rand(rng) < 0.6 ? rand(rng, 1:3) : rand(rng, 1:5)
            elem = (!isempty(prior) && rand(rng) < 0.18) ? RecT(prior[rand(rng, 1:length(prior))]) :
                   Prim(_rand_prim(rng; room = max(room ÷ k, 1)))
            elem isa RecT && (k = min(k, 2))      # keeps leaf counts bounded
            push!(rec.members, Member(member_name!(n, localset, rng, "m$i"), ArrT(elem, k), -1))
        elseif x < 0.49
            push!(rec.members, Member("", RecT(_gen_anon!(u, rng, n, localset)), -1))
        else
            push!(rec.members, Member(member_name!(n, localset, rng, "m$i"), Prim(_rand_prim(rng; room = room)), -1))
        end
    end
    # a record of nothing but padding bitfields has no leaves; give it one
    isempty(leaves(u, me)) && push!(rec.members, Member(member_name!(n, localset, rng, "m98"), Prim(:short), -1))
    if isunion
        # the active member must contribute a leaf
        rec.active = rand(rng, 1:length(rec.members))
        for _ in 1:length(rec.members)
            m = rec.members[rec.active]
            (m.bits == 0 || (m.bits > 0 && isempty(m.name))) || break
            rec.active = mod1(rec.active + 1, length(rec.members))
        end
    end
    style = rand(rng)
    if style < (n.ident ? 0.35 : 0.12)          # struct tag {...};
        rec.tagonly = true
        rec.tag = fresh!(n, rng, "T$me")
        rec.name = rec.tag
    elseif style < 0.55                         # typedef struct {...} name;
        rec.name = fresh!(n, rng, "R$me")
    elseif style < 0.8                          # typedef struct name {...} name;
        rec.name = fresh!(n, rng, "R$me")
        rec.tag = rec.name
    else                                        # typedef struct name_s {...} name;
        rec.name = fresh!(n, rng, "R$me")
        rec.tag = fresh!(n, rng, rec.name * "_s")
    end
    return me
end

function _gen_enum!(u::Unit, rng, n::Names, i::Int)
    k = rand(rng, 2:5)
    style = rand(rng)
    vals = if style < 0.4
        collect(Int64, 0:k-1)
    elseif style < 0.6
        sort!(unique!(Int64.(rand(rng, 1:1000, k))))
    elseif style < 0.75
        collect(Int64, -2:k-3)
    elseif style < 0.85
        Int64[Int64(1) << j for j in 0:k-1]
    else
        # does not fit in 32 bits: GNU C and C++ give the enum a 64-bit type
        Int64[(Int64(1) << 33) + j for j in 0:k-1]
    end
    fixed = :none
    if maximum(vals) < (Int64(1) << 31) && rand(rng) < 0.25
        fixed = minimum(vals) < 0 ? rand(rng, [:schar, :short, :llong]) :
                                    rand(rng, [:uchar, :ushort, :llong, :uint])
        if fixed in (:schar, :uchar) && (maximum(vals) > 127 || minimum(vals) < -128)
            fixed = :llong
        elseif fixed in (:short, :ushort) && (maximum(vals) > 32767 || minimum(vals) < -32768)
            fixed = :llong
        end
    end
    ename = fresh!(n, rng, "E$i")
    tag = rand(rng) < 0.3 ? fresh!(n, rng, ename * "_e") : ""
    enames = [fresh!(n, rng, string("E", i, "_K", j)) for j in 1:length(vals)]
    push!(u.enums, EnumDef(ename, tag, enames, vals, fixed, 0))
end

function _gen_func!(u::Unit, rng, n::Names, id::Int)
    recs = named_records(u)
    es = [k for (k, e) in enumerate(u.enums) if e.twin == 0]
    spill = rand(rng) < 0.22
    np = spill ? rand(rng, 7:13) : rand(rng, 0:4)
    ps = Param[]
    plocal = Set{String}()
    for j in 1:np
        x = rand(rng)
        t = if x < (spill ? 0.25 : 0.42) && !isempty(recs)
            RecT(recs[rand(rng, 1:length(recs))])
        elseif x < (spill ? 0.32 : 0.55) && !isempty(recs)
            ArrT(RecT(recs[rand(rng, 1:length(recs))]), 0)
        elseif x < 0.94 || isempty(es)
            spill ? Prim(rand(rng, [:int, :double, :llong, :float, :char, :short, :ptr, :double])) :
                    Prim(_rand_prim(rng))
        else
            EnumT(es[rand(rng, 1:length(es))])
        end
        push!(ps, Param(member_name!(n, plocal, rng, "p$j"), t))
    end
    if rand(rng) < 0.04
        push!(ps, Param(member_name!(n, plocal, rng, "p$(np + 1)"), Prim(:ldouble)))
    end
    x = rand(rng)
    ret = if x < 0.02
        Prim(:ldouble)
    elseif x < 0.5 && !isempty(recs)
        RecT(recs[rand(rng, 1:length(recs))])
    elseif x < 0.76
        Prim(_rand_prim(rng))
    elseif x < 0.84 && !isempty(es)
        EnumT(es[rand(rng, 1:length(es))])
    else
        nothing
    end
    push!(u.funcs, Func(fresh!(n, rng, "f$id"), ps, ret, id, 1))
end

"""
    generate(seed; ident=false) -> Unit

One random translation unit. Same seed, same unit, on every platform: the
platform only enters through the C compiler's answers, which are read back from
the built library rather than predicted here (the one exception is the width of
`long`, which the return literals are written at; see `lit_width`).
"""
function generate(seed::Integer; ident::Bool = false)
    rng = Random.Xoshiro(seed + (ident ? 0x5eed_0000 : 0))
    n = Names(Set{String}(), Set{String}(), ident)
    u = Unit(Int(seed), ident, Record[], EnumDef[], Func[])
    for i in 1:rand(rng, 1:3)
        _gen_enum!(u, rng, n, i)
    end
    # The identical-member-set pair. C forbids one enumerator twice in a unit,
    # so the second enum lives in a companion translation unit: two DWARF enum
    # types with one member set, the shape `const Alias = Canonical` is for.
    tw = Int64[3, 7, 11]
    push!(u.enums, EnumDef("TwinA", "", ["AF_TW0", "AF_TW1", "AF_TW2"], copy(tw), :none, 1))
    push!(u.enums, EnumDef("TwinB", "", ["AF_TW0", "AF_TW1", "AF_TW2"], copy(tw), :none, 2))
    for _ in 1:rand(rng, 2:6)
        _gen_record!(u, rng, n)
    end
    nf = rand(rng, 3:7)
    for id in 1:nf
        _gen_func!(u, rng, n, id)
    end
    # every named record crosses at least one call by value
    seen = Set{Int}()
    for f in u.funcs
        f.ret isa RecT && push!(seen, f.ret.r)
        for p in f.params
            p.ty isa RecT && push!(seen, p.ty.r)
        end
    end
    for r in named_records(u)
        r in seen && continue
        f = u.funcs[rand(rng, 1:length(u.funcs))]
        if !(f.ret isa RecT) && rand(rng) < 0.5
            f.ret = RecT(r)
        else
            plocal = Set{String}(p.name for p in f.params)
            push!(f.params, Param(member_name!(n, plocal, rng, "p$(length(f.params) + 1)"), RecT(r)))
        end
    end
    if ident
        # A function and a tag-only record may share a spelling in C and C++
        # (tags are their own namespace; a function hides a class name).
        tags = [r for r in named_records(u) if u.records[r].tagonly]
        if !isempty(tags) && rand(rng) < 0.7
            nm = u.records[tags[rand(rng, 1:length(tags))]].tag
            f = u.funcs[rand(rng, 1:length(u.funcs))]
            nm in n.members || (f.name = nm)
        end
        # A zero-argument C function named like the wrapper's module initializer.
        z = [f for f in u.funcs if isempty(f.params)]
        if !isempty(z) && !("__init__" in n.ordinary) && !("__init__" in n.members) && rand(rng) < 0.5
            z[1].name = "__init__"
            push!(n.ordinary, "__init__")
        end
    end
    id = nf
    nreg = length(u.enums) - 2
    push!(u.funcs, Func("af_twin_a", [Param("x", EnumT(nreg + 1)), Param("k", Prim(:int))],
                        EnumT(nreg + 1), id += 1, 1))
    push!(u.funcs, Func("af_twin_b", [Param("x", EnumT(nreg + 2)), Param("k", Prim(:int))],
                        EnumT(nreg + 2), id += 1, 2))
    return u
end

# ── Planned checks ───────────────────────────────────────────────────────────

"""
The checks a unit plans before anything is built. A run asserts that every one
comes back with a verdict, so a check that silently did not run cannot pass.
"""
function planned_checks(u::Unit)
    out = String["load"]
    for r in named_records(u)
        push!(out, "layout/" * u.records[r].name)
        isempty(leaves(u, r)) || push!(out, "access/" * u.records[r].name)
    end
    for f in u.funcs
        isempty(f.params) || push!(out, "call/" * f.name)
        f.ret === nothing || push!(out, "ret/" * f.name)
    end
    push!(out, "exit")
    return out
end

# ── C emission ───────────────────────────────────────────────────────────────

function _decl(u::Unit, m::Member, indent::String)
    isempty(m.name) && m.bits < 0 && return _record_body(u, (m.ty::RecT).r, indent) * ";"
    t = m.ty
    t isa ArrT && return string(type_spell(u, t.elem), " ", m.name, "[", t.n, "];")
    m.bits >= 0 && return string(type_spell(u, t), isempty(m.name) ? "" : " " * m.name, " : ", m.bits, ";")
    return string(type_spell(u, t), " ", m.name, ";")
end

function _record_body(u::Unit, r::Int, indent::String)
    rec = u.records[r]
    io = IOBuffer()
    print(io, rec.isunion ? "union" : "struct")
    isempty(rec.tag) || print(io, " ", rec.tag)
    println(io, " {")
    for m in rec.members
        println(io, indent, "    ", _decl(u, m, indent * "    "))
    end
    print(io, indent, "}")
    rec.packed && print(io, " __attribute__((packed))")
    return String(take!(io))
end

function _emit_record(io, u::Unit, r::Int)
    rec = u.records[r]
    rec.pack > 0 && println(io, "#pragma pack(push, ", rec.pack, ")")
    if rec.tagonly
        println(io, _record_body(u, r, ""), ";")
    else
        println(io, "typedef ", _record_body(u, r, ""), " ", rec.name, ";")
    end
    rec.pack > 0 && println(io, "#pragma pack(pop)")
    println(io)
end

function _int_lit(v::Integer, signed::Bool)
    v = Int128(v)
    if signed
        v >= 0 && return string(v, v > typemax(Int32) ? "LL" : "")
        v == -(Int128(1) << 63) && return "(-9223372036854775807LL - 1)"
        return string("(", v, v < typemin(Int32) ? "LL" : "", ")")
    end
    return string("0x", string(UInt64(mod(v, Int128(1) << 64)); base = 16), "ULL")
end

function _emit_enum(io, u::Unit, e::EnumDef)
    fx = e.fixed === :none ? "" : " : " * PRIM_C[e.fixed]
    body = join(("    $(nm) = $(_int_lit(v, true))" for (nm, v) in zip(e.names, e.values)), ",\n")
    println(io, "typedef enum", isempty(e.tag) ? "" : " " * e.tag, fx, " {\n", body, "\n} ", e.name, ";\n")
end

"""
    emit_enum(io, u, e)
    emit_record(io, u, r)

One enum, or the named record `u.records[r]` with its `#pragma pack`, as
`emit_main` writes them. For a caller that wants the unit's types without its
harness: test/test_sysv_abi_oracle.jl builds its corpus from them.
"""
emit_enum(io, u::Unit, e::EnumDef) = _emit_enum(io, u, e)
emit_record(io, u::Unit, r::Int) = _emit_record(io, u, r)

# A value as a C expression of type `ty`, for the return formulas.
function _c_value(u::Unit, ty::CT, v)
    ty isa EnumT && return u.enums[ty.e].names[enum_index(u, ty.e, v)]
    k = (ty::Prim).k
    k == :float && return @sprintf("%af", Float32(v))
    k == :double && return @sprintf("%a", Float64(v))
    k == :ldouble && return "(long double)" * @sprintf("%a", Float64(v))
    k == :ptr && return "(void*)(uintptr_t)" * _int_lit(v, false)
    k == :bool && return v == 0 ? "0" : "1"
    return "(" * PRIM_C[k] * ")" * _int_lit(v, prim_signed(k))
end

function _rec_macro(ty::CT)
    ty isa EnumT && return "AF_RS"
    k = (ty::Prim).k
    k == :float && return "AF_RF"
    k == :double && return "AF_RD"
    k == :ldouble && return "AF_RL"
    k == :ptr && return "AF_RP"
    k == :char && return "AF_RC"
    return prim_signed(k) ? "AF_RS" : "AF_RU"
end

const _PREAMBLE = """
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#  define AF_BOOL bool
#  define AF_C_BEGIN extern "C" {
#  define AF_C_END }
#else
#  define AF_BOOL _Bool
#  define AF_C_BEGIN
#  define AF_C_END
#endif

/* What a callee read, as C reads it: integers widened by their own signedness
   (plain char by the platform's), floats and pointers by their bits. */
#define AF_RS(i, e) (af_rec[i] = (unsigned long long)(long long)(e))
#define AF_RU(i, e) (af_rec[i] = (unsigned long long)(e))
#define AF_RC(i, e) (af_rec[i] = ((char)-1 < 0) ? (unsigned long long)(long long)(e) : (unsigned long long)(e))
#define AF_RF(i, e) do { float af_f_ = (e); unsigned int af_u_; memcpy(&af_u_, &af_f_, 4); af_rec[i] = af_u_; } while (0)
#define AF_RD(i, e) do { double af_d_ = (e); memcpy(&af_rec[i], &af_d_, 8); } while (0)
#define AF_RP(i, e) (af_rec[i] = (unsigned long long)(uintptr_t)(e))
#define AF_RL(i, e) do { double af_d_ = (double)(e); memcpy(&af_rec[i], &af_d_, 8); } while (0)
"""

function _emit_func(io, u::Unit, f::Func)
    rs = f.ret === nothing ? "void" : type_spell(u, f.ret)
    ps = isempty(f.params) ? "void" : join((isptrparam(p.ty) ?
        string("const ", rec_spell(u, ((p.ty::ArrT).elem::RecT).r), " *", p.name) :
        string(type_spell(u, p.ty), " ", p.name) for p in f.params), ", ")
    println(io, rs, " ", f.name, "(", ps, ")\n{")
    println(io, "    af_rec_fn = ", f.id, ";")
    for s in call_slots(u, f)
        p = f.params[s.param]
        expr, ty = s.leaf === nothing ? (p.name, p.ty) :
            (string(p.name, isptrparam(p.ty) ? "->" : ".", s.leaf.cpath[2:end]), s.leaf.ty)
        println(io, "    ", _rec_macro(ty), "(", s.ord, ", ", expr, ");")
    end
    println(io, "    af_rec_n = ", length(call_slots(u, f)), ";")
    if f.ret isa RecT
        println(io, "    {\n        ", rs, " af_r;\n        memset(&af_r, 0, sizeof af_r);")
        for (ord, l) in enumerate(leaves(u, f.ret.r))
            v = ret_value(u, f, ord - 1, l.ty, l.bits, l.signed, lit_width(l.ty))
            rhs = (l.bits > 0 && l.ty isa Prim && l.ty.k != :bool) ?
                  (v < 0 ? "(" * string(v) * ")" : string(v)) : _c_value(u, l.ty, v)
            println(io, "        af_r", l.cpath, " = ", rhs, ";")
        end
        println(io, "        return af_r;\n    }")
    elseif f.ret !== nothing
        sg = f.ret isa EnumT || prim_signed((f.ret::Prim).k)
        println(io, "    return ", _c_value(u, f.ret, ret_value(u, f, 0, f.ret, -1, sg, lit_width(f.ret))), ";")
    end
    println(io, "}\n")
end

"Source file names for a unit in `lang` (:c or :cpp): main and companion."
function src_names(u::Unit, lang::Symbol)
    ext = lang === :cpp ? ".cpp" : ".c"
    return ("af" * unit_tag(u) * ext, "af" * unit_tag(u) * "_twin" * ext)
end

"""
    emit_main(u) -> String

The main translation unit: the API under test, the recorder, and the table the
child reads through `cglobal`:

  af_meta   magic, #records, #paths, #bitfield probes, #prims, probe stride,
            #enums, char signedness
  af_prims  sizeof each `PRIM_ORDER` kind
  af_esizes sizeof each enum of the main unit
  af_sizes  sizeof each named record (model order)
  af_offs   offsetof each `paths` entry of each named record, in order
  af_bfp    per named bitfield, an image of its record with only that field
            all-ones. A load-time initializer writes them: C has no offsetof
            for a bitfield, and C++ has no designated initializers to do it
            statically. `af_ready` says the initializer ran.
"""
function emit_main(u::Unit)
    io = IOBuffer()
    println(io, "/* abi_fuzz unit ", unit_tag(u), ", generated by test/tools/abi_fuzz.jl.")
    println(io, "   Compiled as C (Tier 3) and, unchanged, as C++ (Tier 2). */")
    print(io, _PREAMBLE)
    println(io, "\nAF_C_BEGIN\nunsigned long long af_rec[", max_slots(u) + 8,
            "];\nint af_rec_n;\nint af_rec_fn;\nAF_C_END\n")
    for e in u.enums
        e.twin == 2 || _emit_enum(io, u, e)
    end
    for (r, rec) in enumerate(u.records)
        rec.anon || _emit_record(io, u, r)
    end
    for f in u.funcs
        f.tu == 1 && _emit_func(io, u, f)
    end
    recs = named_records(u)
    probes = bitfield_probes(u)
    offs = [string("offsetof(", rec_spell(u, r), ", ", p.cpath[2:end], ")") for r in recs for p in paths(u, r)]
    ens = [e for e in u.enums if e.twin != 2]
    println(io, "union af_all {\n    char af_c;")
    for (i, r) in enumerate(recs)
        println(io, "    ", rec_spell(u, r), " af_r", i, ";")
    end
    println(io, "};\n")
    println(io, "AF_C_BEGIN")
    println(io, "unsigned long long af_meta[] = { 0xAFAB1F00ULL, ", length(recs), ", ", length(offs), ", ",
            length(probes), ", ", length(PRIM_ORDER), ", sizeof(union af_all), ", length(ens), ", (char)-1 < 0 };")
    println(io, "unsigned long long af_prims[] = { ",
            join(("sizeof(" * PRIM_C[k] * ")" for k in PRIM_ORDER), ", "), " };")
    println(io, "unsigned long long af_esizes[] = { ", join(("sizeof(" * e.name * ")" for e in ens), ", "), " };")
    println(io, "unsigned long long af_sizes[] = { ",
            isempty(recs) ? "0" : join(("sizeof(" * rec_spell(u, r) * ")" for r in recs), ", "), " };")
    println(io, "unsigned long long af_offs[] = {\n    ", isempty(offs) ? "0" : join(offs, ",\n    "), " };")
    println(io, "unsigned char af_bfp[", max(length(probes), 1), "][sizeof(union af_all)];")
    println(io, "unsigned long long af_ready;")
    println(io, "AF_C_END\n")
    println(io, "static void af_init_bfp(void)\n{")
    for (i, (r, l)) in enumerate(probes)
        one = l.ty isa Prim && l.ty.k == :bool ? "1" : "-1"
        println(io, "    { ", rec_spell(u, r), " s; memset(&s, 0, sizeof s); s", l.cpath, " = ", one,
                "; memcpy(af_bfp[", i - 1, "], &s, sizeof s); }")
    end
    println(io, "    af_ready = 0xAF5EEDULL;\n}\n")
    println(io, "#ifdef __cplusplus\nstatic int af_bfp_init_ = (af_init_bfp(), 1);\n#else\n",
            "__attribute__((constructor)) static void af_bfp_ctor_(void) { af_init_bfp(); }\n#endif")
    return String(take!(io))
end

"The companion unit: the second enum of the twin pair and its one function."
function emit_twin(u::Unit)
    io = IOBuffer()
    println(io, "/* abi_fuzz unit ", unit_tag(u), ", companion: the identical-member-set enum. */")
    print(io, _PREAMBLE)
    println(io, "\nAF_C_BEGIN\nextern unsigned long long af_rec[];\nextern int af_rec_n;\nextern int af_rec_fn;\nAF_C_END\n")
    for e in u.enums
        e.twin == 2 && _emit_enum(io, u, e)
    end
    for f in u.funcs
        f.tu == 2 && _emit_func(io, u, f)
    end
    return String(take!(io))
end

end # module AbiFuzzModel
