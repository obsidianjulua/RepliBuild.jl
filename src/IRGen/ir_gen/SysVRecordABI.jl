# SysVRecordABI.jl — how a by-value record crosses under x86-64 SysV, from its
# DWARF facts. Included by FunctionGen.jl, whose lookups (`_table_entry`,
# `_fuzzy_struct_lookup`, `_strip_top_cv`) it shares.
#
# The dialect used to classify a record from the LLVM body StructGen emits for
# it, and a body cannot carry what the convention asks: an explicit pad array is
# bytes like any other (an eightbyte of a float and padding went in a GPR), a
# tiled body spells a nested record, an enum or a union's members as bytes (a
# misaligned field inside went unseen, a packed record went in registers), a
# bitfield's storage unit sits at its byte offset with its whole type (which
# read as misaligned where C checks a bitfield's bits only), `long double` and
# `_Complex` were `!llvm.ptr`, and no body says `aligned(16)`. The facts are in
# the metadata: members at their offsets, bitfields by bit range, nested
# records, unions, the scalar kinds, DW_AT_alignment. This file classifies from
# them, FunctionGen states the verdict on each `jlcs.ffe_call`/`jlcs.try_call`
# (`sysv_args`, `sysv_ret`), and `buildCallShape` in JLCSPasses.cpp acts on it
# under the SysV target. The Win64 rules never read it: they cross a record by
# size alone.
#
# The rules are the ABI's as clang's X86_64ABIInfo and LLVM's llvm/ABI X86.cpp
# implement them (`test_sysv_abi_oracle.jl` diffs this against the latter):
#   * every eightbyte starts NO_CLASS; only fields classify, never padding;
#   * a nested record contributes its fields at their real offsets; an array
#     each element; a union every member at offset 0;
#   * a field not at a multiple of its type's alignment makes the record
#     MEMORY, except a bitfield, which classifies by its bit range with no
#     alignment check, and an unnamed bitfield, which is padding;
#   * integers, pointers and enums are INTEGER; `float`, `double`, `_Float16`
#     and both halves of a `_Complex` are SSE; `long double` is X87 with X87UP
#     above it; `__int128` is INTEGER in both eightbytes;
#   * merge: equal stays, NO_CLASS yields, MEMORY wins, then INTEGER, then any
#     x87 class against another class is MEMORY, else SSE; afterwards an
#     X87UP not under X87 is MEMORY;
#   * over 16 bytes, or with a MEMORY eightbyte, the record is MEMORY; its
#     byval and sret slots take the record's alignment (at least 8), which is
#     DW_AT_alignment where clang wrote it and the members' alignment lowered
#     to what the layout shows otherwise (`__attribute__((packed))` writes none);
#   * X87 as an argument is MEMORY with the record's alignment (16); as a
#     return it is st0, which the thunk does not read, so FunctionGen traps it;
#   * a record whose fields are all NO_CLASS (padding only) is passed nowhere
#     and returned as void.
# A member whose kind cannot be read (an unknown spelling, a record with no
# definition) leaves a record of 16 bytes or less unclassifiable, and
# FunctionGen traps that too, as it did for a byte region that might hold a
# float. Over 16 bytes the class is MEMORY whatever the member is.

"""
    SysVRecordABI

How a record crosses by value under x86-64 SysV (`sysv_record_abi`).

  * `kind`: `:regs` (in registers, one class per eightbyte in `classes`),
    `:memory` (byval / sret, `align`), `:x87` (`long double` alone: an
    argument in memory, a return in st0), `:ignore` (only padding: no
    operand, a void return), `:unknown` (`reason` says what could not be read).
  * `classes`: for `:regs`, `:int` or `:sse` per eightbyte C passes, and
    `:none` for a low eightbyte of padding C passes nowhere (the high one
    then travels alone, read from offset 8).
  * `size`, `align`: the record's bytes and alignment.
"""
struct SysVRecordABI
    kind::Symbol
    classes::Vector{Symbol}
    size::Int
    align::Int
    reason::String
end

# One classified piece of a record: a scalar, a bitfield, or a nested record's
# own position (`:place`, checked for alignment and nothing else).
struct _SysVLeaf
    bit::Int          # absolute bit offset in the record
    bits::Int         # width: a bitfield's declared width, else 8 × size
    kind::Symbol      # :int | :sse | :x87 | :place | :memory | :unknown
    align::Int        # the type's alignment in bytes, for the misalignment check
    bitfield::Bool
    what::String
end

# Integer spellings and the bytes each has when the metadata gives no size (an
# array element). `long` follows the host: 8 on LP64, 4 on LLP64.
const _SYSV_INT_WIDTH = Dict{String,Int}(
    "char" => 1, "signed char" => 1, "unsigned char" => 1, "_Bool" => 1, "bool" => 1, "char8_t" => 1,
    "short" => 2, "short int" => 2, "signed short" => 2, "signed short int" => 2, "unsigned short" => 2,
    "unsigned short int" => 2, "short unsigned int" => 2, "char16_t" => 2,
    "int" => 4, "signed int" => 4, "signed" => 4, "unsigned" => 4, "unsigned int" => 4, "wchar_t" => sizeof(Cwchar_t),
    "char32_t" => 4,
    "long" => sizeof(Clong), "long int" => sizeof(Clong), "signed long" => sizeof(Clong),
    "signed long int" => sizeof(Clong), "unsigned long" => sizeof(Clong), "unsigned long int" => sizeof(Clong),
    "long unsigned int" => sizeof(Clong),
    "long long" => 8, "long long int" => 8, "signed long long" => 8, "signed long long int" => 8,
    "unsigned long long" => 8, "unsigned long long int" => 8, "long long unsigned int" => 8,
    "int8_t" => 1, "uint8_t" => 1, "int16_t" => 2, "uint16_t" => 2, "int32_t" => 4, "uint32_t" => 4,
    "int64_t" => 8, "uint64_t" => 8, "int_least8_t" => 1, "uint_least8_t" => 1, "int_least16_t" => 2,
    "uint_least16_t" => 2, "int_least32_t" => 4, "uint_least32_t" => 4, "int_least64_t" => 8, "uint_least64_t" => 8,
    "int_fast8_t" => 1, "uint_fast8_t" => 1, "int_fast16_t" => 8, "uint_fast16_t" => 8, "int_fast32_t" => 8,
    "uint_fast32_t" => 8, "int_fast64_t" => 8, "uint_fast64_t" => 8,
    "size_t" => 8, "ssize_t" => 8, "ptrdiff_t" => 8, "intptr_t" => 8, "uintptr_t" => 8, "intmax_t" => 8,
    "uintmax_t" => 8, "off_t" => 8, "time_t" => 8,
    "__int128" => 16, "unsigned __int128" => 16, "__int128_t" => 16, "__uint128_t" => 16,
)

# Julia spellings the metadata gives a member whose C spelling is a typedef
# (`typedef float real;` is `real` in `c_type` and `Cfloat` in `julia_type`).
const _SYSV_JULIA_SSE = Dict("Cfloat" => 4, "Float32" => 4, "Cdouble" => 8, "Float64" => 8, "Float16" => 2)
const _SYSV_JULIA_INT = Set(["Bool", "Cchar", "Cuchar", "UInt8", "Int8", "Cshort", "Cushort", "Int16", "UInt16",
                             "Cint", "Cuint", "Int32", "UInt32", "Clong", "Culong", "Clonglong", "Culonglong",
                             "Int64", "UInt64", "Csize_t", "Cssize_t", "Cptrdiff_t", "Cintmax_t", "Cuintmax_t",
                             "Cwchar_t", "Int128", "UInt128", "Int", "UInt", "Cstring", "Cwstring"])

_sysv_norm(t::AbstractString) = String(strip(replace(
    replace(String(t), r"\b(?:const|volatile|restrict|_Atomic)\b" => ""), r"\s+" => " ")))

# The scalar a C spelling names: `(kind, size, align)`, or `nothing`. `size` is
# the member's bytes where the metadata has them (a typedef'd integer is its
# own width), else the spelling's. A scalar's alignment is its size.
function _sysv_scalar(t::AbstractString, size::Int)
    t == "float" && return (:sse, 4, 4)
    t == "double" && return (:sse, 8, 8)
    t in ("_Float16", "__fp16", "__bf16", "half") && return (:sse, 2, 2)
    t == "long double" && return (:x87, 16, 16)
    if haskey(_SYSV_INT_WIDTH, t)
        n = size > 0 ? size : _SYSV_INT_WIDTH[t]
        return (:int, n, n)
    end
    # A spelling the MLIR type map knows that is not listed above (`int32`,
    # `Csize_t`): its width from the type. `f80` and `i128` are listed.
    mapped = map_cpp_type(String(t))
    im = match(r"^i(\d+)$", mapped)
    if im !== nothing
        n = max(cld(parse(Int, im.captures[1]), 8), 1)
        n = size > 0 ? size : n
        return (:int, n, n)
    end
    fm = match(r"^f(16|32|64)$", mapped)
    fm !== nothing && return (:sse, parse(Int, fm.captures[1]) ÷ 8, parse(Int, fm.captures[1]) ÷ 8)
    return nothing
end

# The scalar a Julia spelling names, for a member whose C spelling is a typedef.
function _sysv_julia_scalar(jt::AbstractString, size::Int)
    jt = String(strip(jt))
    haskey(_SYSV_JULIA_SSE, jt) && return (:sse, _SYSV_JULIA_SSE[jt], _SYSV_JULIA_SSE[jt])
    (startswith(jt, "Ptr{") || jt in ("Cstring", "Cwstring")) && return (:int, 8, 8)
    if jt in _SYSV_JULIA_INT
        n = size > 0 ? size : jt in ("Int128", "UInt128") ? 16 : 8
        return (:int, n, n)
    end
    return nothing
end

# Element type and count of a C array spelling, or `nothing`. A dimension with
# no bound (`T[]`, a flexible array member) counts as 0.
function _sysv_array(t::AbstractString)
    m = match(r"^(.*?)\s*((?:\[\d*\])+)$", t)
    (m === nothing || occursin("(", String(m.captures[1]))) && return nothing
    dims = [isempty(d.captures[1]) ? 0 : parse(Int, d.captures[1]) for d in eachmatch(r"\[(\d*)\]", String(m.captures[2]))]
    return (String(strip(m.captures[1])), dims)
end

# The element of an `NTuple{N, T}` Julia spelling, else `"Any"`.
function _sysv_julia_elem(jt::AbstractString)
    m = match(r"^NTuple\{\d+,\s*(.+)\}$", strip(jt))
    return m === nothing ? "Any" : String(m.captures[1])
end

"""
    _sysv_type_leaves!(out, c_type, julia_type, size, bit, structs, seen, what)

The leaves of one member type at absolute bit offset `bit`. Returns the type's
alignment in bytes, or `nothing` when it could not be read (an `:unknown` leaf
was pushed then).
"""
function _sysv_type_leaves!(out::Vector{_SysVLeaf}, c_type::AbstractString, julia_type::AbstractString,
                            size::Int, bit::Int, structs, seen::Set{String}, what::String)
    t = _sysv_norm(c_type)
    t = String(replace(t, r"^(?:struct|union|enum|class) " => ""))
    unknown(why) = (push!(out, _SysVLeaf(bit, 8 * max(size, 0), :unknown, 1, false, "$what: $why")); nothing)

    (isempty(t) || t == "unknown") && return unknown("its type is not in the metadata")
    if endswith(t, "*") || endswith(t, "&") || occursin("(*", t) || occursin("(&", t) || startswith(t, "function_ptr")
        push!(out, _SysVLeaf(bit, 64, :int, 8, false, what))
        return 8
    end

    arr = _sysv_array(t)
    if arr !== nothing
        elem, dims = arr
        if any(==(0), dims)
            push!(out, _SysVLeaf(bit, 0, :memory, 1, false, "$what: an array with no bound"))
            return nothing
        end
        n = prod(dims)
        esize = size > 0 ? size ÷ n : 0
        ejt = _sysv_julia_elem(julia_type)
        # One element gives the alignment and settles the element size.
        probe = _SysVLeaf[]
        al = _sysv_type_leaves!(probe, elem, ejt, esize, bit, structs, seen, "$what[]")
        al === nothing && (append!(out, probe); return nothing)
        if esize == 0
            esize = _sysv_leaf_extent(probe, bit)
            esize == 0 && return unknown("the element size of `$t` is not in the metadata")
        end
        append!(out, probe)
        for i in 1:n-1
            _sysv_type_leaves!(out, elem, ejt, esize, bit + 8 * esize * i, structs, seen, "$what[$i]")
        end
        return al
    end

    sc = _sysv_scalar(t, size)
    if sc !== nothing
        kind, n, al = sc
        push!(out, _SysVLeaf(bit, 8 * n, kind, al, false, what))
        return al
    end
    cm = match(r"^(?:_Complex|complex) (float|double|long double)$", t)
    if cm !== nothing
        kind, n, al = _sysv_scalar(String(cm.captures[1]), 0)
        push!(out, _SysVLeaf(bit, 8 * n, kind, al, false, "$what (real)"))
        push!(out, _SysVLeaf(bit + 8 * n, 8 * n, kind, al, false, "$what (imaginary)"))
        return al
    end

    entry, key = _table_entry(t, structs)
    if entry === :enum
        n = size > 0 ? size : something(_parse_num(get(structs[key], "byte_size", nothing)), 4)
        push!(out, _SysVLeaf(bit, 8 * n, :int, n, false, what))
        return n
    elseif entry === :layout
        key in seen && return unknown("`$key` holds itself by value")
        al = _sysv_record_align(key, structs, seen)
        push!(out, _SysVLeaf(bit, 8 * _sysv_record_size(structs[key]), :place, al, false, what))
        push!(seen, key)
        _sysv_record_leaves!(out, key, structs, bit, seen)
        delete!(seen, key)
        return al
    elseif entry === :direct
        # An STL container: its layout is not in the metadata.
        return unknown("`$t` is a toolchain type whose layout is not in the metadata")
    end

    sc = _sysv_julia_scalar(julia_type, size)
    if sc !== nothing
        kind, n, al = sc
        push!(out, _SysVLeaf(bit, 8 * n, kind, al, false, what))
        return al
    end
    return unknown("`$t` is neither a scalar the classifier knows nor a record in the metadata")
end

# The bytes from `bit` to the end of the last leaf pushed for one element.
function _sysv_leaf_extent(leaves::Vector{_SysVLeaf}, bit::Int)
    hi = maximum((l.bit + l.bits for l in leaves if l.kind !== :place); init = bit)
    return cld(hi - bit, 8)
end

_sysv_record_size(info) = something(_parse_num(get(info, "byte_size", nothing)), 0)

"""
    _sysv_record_leaves!(out, key, structs, base_bit, seen)

The leaves of record `key` placed at `base_bit`: its bases' and members' at
their offsets. A union's members all sit at its start. A member with no
position (a static) is not part of the object; an unnamed bitfield is padding.
"""
function _sysv_record_leaves!(out::Vector{_SysVLeaf}, key::String, structs, base_bit::Int, seen::Set{String})
    info = structs[key]
    is_union = get(info, "kind", "struct") == "union"
    for b in get(info, "base_classes", [])
        bt = String(get(b, "type", ""))
        off = something(_parse_num(get(b, "offset", 0)), 0)
        if get(b, "virtual", false) == true
            push!(out, _SysVLeaf(base_bit, 0, :unknown, 1, false, "$key: a virtual base `$bt`"))
            continue
        end
        _sysv_type_leaves!(out, bt, "Any", 0, base_bit + 8 * off, structs, seen, "$key's base `$bt`")
    end
    for m in get(info, "members", [])
        name = String(get(m, "name", ""))
        bits = _parse_num(get(m, "bit_size", nothing))
        size = something(_parse_num(get(m, "size", 0)), 0)
        if bits !== nothing
            isempty(name) && continue                 # an unnamed bitfield is padding
            at = _parse_num(get(m, "data_bit_offset", nothing))
            at === nothing && !is_union &&
                (push!(out, _SysVLeaf(base_bit, bits, :unknown, 1, true, "$key.$name: a bitfield with no DW_AT_data_bit_offset")); continue)
            push!(out, _SysVLeaf(base_bit + something(at, 0), bits, :int, 1, true, "$key.$name"))
            continue
        end
        off = _parse_num(get(m, "offset", nothing))
        if off === nothing
            is_union || continue
            off = 0
        end
        _sysv_type_leaves!(out, String(get(m, "c_type", "")), String(get(m, "julia_type", "Any")), size,
                           base_bit + 8 * off, structs, seen, "$key.$name")
    end
    return out
end

# The alignment of one member type, or `nothing`: the same derivation as the
# leaves, read off a scratch list.
function _sysv_type_align(c_type, julia_type, size::Int, structs, seen::Set{String})
    scratch = _SysVLeaf[]
    return _sysv_type_leaves!(scratch, c_type, julia_type, size, 0, structs, seen, "")
end

"""
    _sysv_record_align(key, structs, seen) -> Int

The record's alignment: DW_AT_alignment where clang wrote it (it writes the
actual alignment whenever `aligned` or `#pragma pack` applies, and nothing for
a plain or `__attribute__((packed))` record), else the largest of its members'
and bases' alignments, lowered until the layout fits it: the size a multiple
and every placed member at a multiple of the smaller of it and the member's own.
`packed { char; long double; }` is 17 bytes, so 16 lowers to 1, which is what
clang gives it, and its byval slot is 8-aligned like clang's.
"""
function _sysv_record_align(key::String, structs, seen::Set{String} = Set{String}())
    info = structs[key]
    explicit = _parse_num(get(info, "alignment", nothing))
    explicit !== nothing && explicit > 0 && return explicit
    size = _sysv_record_size(info)
    is_union = get(info, "kind", "struct") == "union"
    placed = Tuple{Int,Int}[]                       # (byte offset, type alignment)
    al = 1
    for b in get(info, "base_classes", [])
        a = _sysv_type_align(String(get(b, "type", "")), "Any", 0, structs, seen)
        a === nothing && continue
        al = max(al, a)
        push!(placed, (something(_parse_num(get(b, "offset", 0)), 0), a))
    end
    for m in get(info, "members", [])
        ma = _parse_num(get(m, "alignment", nothing))
        ma === nothing || (al = max(al, ma))
        haskey(m, "bit_size") && continue             # a bitfield's unit is placed by its type, which C does not check
        off = _parse_num(get(m, "offset", nothing))
        off === nothing && (is_union || continue; off = 0)
        a = _sysv_type_align(String(get(m, "c_type", "")), String(get(m, "julia_type", "Any")),
                             something(_parse_num(get(m, "size", 0)), 0), structs, seen)
        a === nothing && continue
        al = max(al, a)
        push!(placed, (off, a))
    end
    fits(a) = (size == 0 || size % a == 0) && all(((off, ta),) -> off % min(a, ta) == 0, placed)
    while al > 1 && !fits(al)
        al ÷= 2
    end
    return al
end

# Rule 4's merge of a field's class into an eightbyte's.
function _sysv_merge(acc::Symbol, field::Symbol)
    (acc == field || field == :none) && return acc
    field == :memory && return :memory
    acc == :none && return field
    (acc == :int || field == :int) && return :int
    (field in (:x87, :x87up) || acc in (:x87, :x87up)) && return :memory
    return :sse
end

"""
    sysv_record_abi(key, structs) -> SysVRecordABI

How the record `structs[key]` crosses by value under x86-64 SysV; see the head
of this file for the rules.
"""
function sysv_record_abi(key::AbstractString, structs)
    key = String(key)
    info = structs[key]
    size = _sysv_record_size(info)
    align = _sysv_record_align(key, structs)
    size == 0 && return SysVRecordABI(:ignore, Symbol[], 0, align, "")
    size > 16 && return SysVRecordABI(:memory, Symbol[], size, align, "")

    leaves = _SysVLeaf[]
    _sysv_record_leaves!(leaves, key, structs, 0, Set{String}([key]))
    lo = hi = :none
    for lf in leaves
        lf.kind === :memory && return SysVRecordABI(:memory, Symbol[], size, align, "")
        lf.kind === :unknown && return SysVRecordABI(:unknown, Symbol[], size, align, lf.what)
        if !lf.bitfield && lf.bit % (8 * lf.align) != 0
            return SysVRecordABI(:memory, Symbol[], size, align, "")
        end
        lf.kind === :place && continue
        if lf.kind === :x87
            lo = _sysv_merge(lo, :x87)
            hi = _sysv_merge(hi, :x87up)
        else
            lf.bits == 0 && continue
            c = lf.kind === :sse ? :sse : :int
            eb_lo = lf.bit ÷ 64
            eb_hi = (lf.bit + lf.bits - 1) ÷ 64
            eb_hi > 1 && return SysVRecordABI(:unknown, Symbol[], size, align,
                                              "$(lf.what) ends past the record's 16 bytes")
            eb_lo == 0 && (lo = _sysv_merge(lo, c))
            eb_hi == 1 && (hi = _sysv_merge(hi, c))
        end
        (lo === :memory || hi === :memory) && break
    end
    # Rule 5: the post-merger cleanup.
    hi === :memory && (lo = :memory)
    hi === :x87up && lo !== :x87 && (lo = :memory)
    lo === :memory && return SysVRecordABI(:memory, Symbol[], size, align, "")
    lo === :x87 && return SysVRecordABI(:x87, Symbol[], size, align, "")
    (lo === :none && hi === :none) && return SysVRecordABI(:ignore, Symbol[], size, align, "")
    classes = Symbol[lo]
    hi in (:int, :sse) && push!(classes, hi)
    return SysVRecordABI(:regs, classes, size, align, "")
end

"""
    sysv_call_fact(abi::SysVRecordABI) -> String

The statement FunctionGen puts on a call for a record crossing, which
`buildCallShape` reads: `mem:<align>` (the byval or sret alignment, at least 8),
`regs:<class>[,<class>]` (`int`, `sse`, or `none` for a low eightbyte of
padding), or `ignore`. An `:x87` record is `mem:16` as an argument; as a return
it is a trap (`sysv_trap_reason`), never a statement. `:unknown` is a trap too.
"""
function sysv_call_fact(abi::SysVRecordABI)::String
    abi.kind === :ignore && return "ignore"
    abi.kind in (:memory, :x87) && return "mem:$(max(abi.align, 8))"
    abi.kind === :regs && return "regs:" * join(string.(abi.classes), ",")
    throw(ArgumentError("a `$(abi.kind)` record crossing has no call statement: $(abi.reason)"))
end

"""
    sysv_trap_reason(key, structs; ret, abi = host) -> Union{String,Nothing}

Why the thunk generator refuses to pass (`ret = false`) or return (`ret = true`)
the record `key` by value under `abi`, or `nothing`. SysV refuses a record of
16 bytes or less whose class it cannot read from DWARF, and a `long double`
record as a return, which C leaves in st0 where no thunk reads. Win64 crosses a
record by size alone, so it refuses nothing here.
"""
function sysv_trap_reason(key, structs; ret::Bool, abi::Symbol = Sys.iswindows() ? :win64 : :sysv)
    abi === :win64 && return nothing
    abi === :sysv || throw(ArgumentError("unknown record ABI `$abi`; expected :sysv or :win64"))
    (key === nothing || !haskey(structs, key)) && return nothing
    v = sysv_record_abi(key, structs)
    v.kind === :unknown &&
        return "whose SysV class cannot be read from DWARF ($(v.reason))"
    ret && v.kind === :x87 &&
        return "which SysV returns on the x87 stack (st0), where a thunk does not look"
    return nothing
end
