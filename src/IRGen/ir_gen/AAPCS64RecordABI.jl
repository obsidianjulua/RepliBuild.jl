# AAPCS64RecordABI.jl — how a by-value record crosses under AArch64 AAPCS64
# (Linux), from its DWARF facts. Included by FunctionGen.jl after
# SysVRecordABI.jl, whose leaf walk (`_sysv_record_leaves!`), record alignment
# (`_sysv_record_align`) and vector detection (`_sysv_holds_vector`) it reads:
# the facts are the same, only the rules differ. FunctionGen states the
# verdict on each `jlcs.ffe_call`/`jlcs.try_call` (`aapcs64_args`,
# `aapcs64_ret`) when it generates for an AAPCS64 host, and `buildCallShape`
# in JLCSPasses.cpp acts on it under `AbiTarget::AAPCS64`.
#
# The rules are the ABI's as clang's AArch64ABIInfo implements them for
# `aarch64-linux-gnu` (test_aapcs64_abi_spec.jl diffs this against clang's
# lowering of the same records, and test_aapcs64_record_abi.jl pins one record
# per rule):
#   * a record of 0 bytes is passed nowhere and returned as void;
#   * a homogeneous floating-point aggregate (every scalar leaf the same
#     floating type, at most four of them, no padding: leaves × width = size;
#     nested records and arrays flattened, a union's members counted once per
#     position, an empty base skipped, a bitfield disqualifying) travels in one
#     floating register per leaf, `[N x fT]`, at ANY size: four doubles are 32
#     bytes and still direct, both ways;
#   * otherwise a record of 16 bytes or less travels in general registers. As
#     an argument it is `i64` or `[2 x i64]` by size, or `i128` (an even
#     register pair) when the record's UNADJUSTED alignment is 16: the largest
#     alignment of its members, counting a member's own `alignas` and a nested
#     record's `aligned`, but not an `aligned` on the record itself. As a
#     return it is `iN` of its exact size up to 8 bytes, then `[2 x i64]`, or
#     `i128` when its ADJUSTED alignment (the record's own attribute counts) is
#     16. Field alignment never matters: a packed `{ char; double }` is
#     `[2 x i64]`, and so is `{ float; float; char }`, whose padding breaks the
#     HFA;
#   * over 16 bytes, and not an HFA, the caller copies the record and passes a
#     pointer, with no `byval`; a return goes through a hidden pointer (sret).
#     Both carry the record's alignment;
#   * a record with no scalar of its own but a size (a C++ empty class, a
#     padding-only `{ int : 32; }`) is passed as one `i64` and returned as void:
#     clang's rule for C++, which is what the thunks serve. (C drops the
#     argument too; a C translation unit inside a C++ library is not told apart.)
# Not modelled: a record holding a SIMD vector, which AAPCS64 may pass as a
# homogeneous vector aggregate in Q registers; it is refused. Apple's arm64 PCS
# (empty records dropped in C++, variadic arguments on the stack) is not this
# file; RepliBuild refuses macOS at load.
# A member whose kind cannot be read leaves a record of 64 bytes or less (the
# largest HFA: four `long double`s) unclassifiable, refused as SysV refuses it.

"""
    AAPCS64RecordABI

How a record crosses by value under AArch64 AAPCS64 (`aapcs64_record_abi`).

  * `kind`: `:hfa` (one floating register per leaf: `base` × `count`), `:gpr`
    (`count` general registers, 1 or 2: `i64` or `[2 x i64]`), `:i128` (an
    even general register pair), `:int` (a return of `count` bits in one
    register), `:memory` (a pointer to a copy, or an sret slot, of `align`),
    `:ignore` (0 bytes: no operand, a void return), `:unknown` (`reason` says
    what could not be read).
  * `size`, `align`: the record's bytes and its adjusted alignment.
"""
struct AAPCS64RecordABI
    kind::Symbol
    base::Symbol
    count::Int
    size::Int
    align::Int
    reason::String
end

# The HFA base type a leaf width names, in bits.
const _A64_HFA_BASE = Dict(16 => :f16, 32 => :f32, 64 => :f64, 128 => :f128)

# `(base, count)` when the scalar leaves form a homogeneous floating-point
# aggregate of `size` bytes, else `nothing`. A `long double` leaf is `:x87` in
# the SysV walk's vocabulary and `f128` here.
function _a64_hfa(scalars::Vector{_SysVLeaf}, size::Int)
    isempty(scalars) && return nothing
    all(lf -> lf.kind in (:sse, :x87) && !lf.bitfield && lf.bits == scalars[1].bits, scalars) || return nothing
    bits = scalars[1].bits
    base = get(_A64_HFA_BASE, bits, nothing)
    base === nothing && return nothing
    at = sort!(unique([lf.bit for lf in scalars]))      # a union's members share a position
    n = length(at)
    (n <= 4 && n * bits == 8 * size && at == [i * bits for i in 0:n-1]) || return nothing
    return (base, n)
end

"""
    aapcs64_record_abi(key, structs; ret) -> AAPCS64RecordABI

How the record `structs[key]` crosses by value under AAPCS64, as an argument
(`ret = false`) or a return (`ret = true`); see the head of this file for the
rules. The two differ for a record of 9 to 16 bytes with an `aligned` attribute
of its own, and for a record with no scalar of its own.
"""
function aapcs64_record_abi(key::AbstractString, structs; ret::Bool)
    key = String(key)
    info = structs[key]
    size = _sysv_record_size(info)
    align = _sysv_record_align(key, structs)
    size == 0 && return AAPCS64RecordABI(:ignore, :none, 0, 0, align, "")
    if _sysv_holds_vector(key, structs, Set{String}())
        return AAPCS64RecordABI(:unknown, :none, 0, size, align,
            "holds a SIMD vector, which AAPCS64 may pass as a homogeneous vector aggregate in " *
            "vector registers, and the thunk does not model that")
    end
    leaves = _SysVLeaf[]
    _sysv_record_leaves!(leaves, key, structs, 0, Set{String}([key]); vectors = false)
    scalars = [lf for lf in leaves if lf.kind !== :place]
    for lf in scalars
        lf.kind in (:unknown, :unstated) && size <= 64 &&
            return AAPCS64RecordABI(:unknown, :none, 0, size, align, lf.what)
    end
    hfa = _a64_hfa(scalars, size)
    hfa !== nothing && return AAPCS64RecordABI(:hfa, hfa[1], hfa[2], size, align, "")
    size > 16 && return AAPCS64RecordABI(:memory, :none, 0, size, align, "")
    if ret
        isempty(scalars) && return AAPCS64RecordABI(:ignore, :none, 0, size, align, "")
        size <= 8 && return AAPCS64RecordABI(:int, :none, 8 * size, size, align, "")
        align >= 16 && return AAPCS64RecordABI(:i128, :none, 0, size, align, "")
        return AAPCS64RecordABI(:gpr, :none, 2, size, align, "")
    end
    _sysv_record_align(key, structs; own = false) >= 16 &&
        return AAPCS64RecordABI(:i128, :none, 0, size, align, "")
    return AAPCS64RecordABI(:gpr, :none, cld(size, 8), size, align, "")
end

"""
    aapcs64_call_fact(abi::AAPCS64RecordABI) -> String

The statement FunctionGen puts on a call for a record crossing under AAPCS64,
which `buildCallShape` reads: `ignore`, `mem:<align>` (a pointer to a copy, or
an sret slot, of that alignment, at least 8), `gpr:<n>` (`i64` or `[2 x i64]`),
`i128`, `int:<bits>` (a return in one register), or `hfa:<f16|f32|f64|f128>:<n>`
(`[n x fT]`). `:unknown` is a trap (`aapcs64_trap_reason`), never a statement.
"""
function aapcs64_call_fact(abi::AAPCS64RecordABI)::String
    abi.kind === :ignore && return "ignore"
    abi.kind === :memory && return "mem:$(max(abi.align, 8))"
    abi.kind === :gpr && return "gpr:$(abi.count)"
    abi.kind === :i128 && return "i128"
    abi.kind === :int && return "int:$(abi.count)"
    abi.kind === :hfa && return "hfa:$(abi.base):$(abi.count)"
    throw(ArgumentError("a `$(abi.kind)` record crossing has no call statement: $(abi.reason)"))
end

"""
    aapcs64_statement(key, structs; ret) -> String

What the thunk generator states on a call for the record `key` crossing by
value under AAPCS64: `aapcs64_call_fact` of its class, or `""` for a record
whose class cannot be read (which `aapcs64_trap_reason` refuses first).
"""
function aapcs64_statement(key, structs; ret::Bool)::String
    abi = aapcs64_record_abi(key, structs; ret)
    return abi.kind === :unknown ? "" : aapcs64_call_fact(abi)
end

"""
    aapcs64_trap_reason(key, structs; ret) -> Union{String,Nothing}

Why the thunk generator refuses to pass or return the record `key` by value
under AAPCS64, or `nothing`: a class DWARF does not settle (a member whose kind
cannot be read, a SIMD vector member). `sysv_trap_reason` dispatches here for
`abi = :aapcs64`.
"""
function aapcs64_trap_reason(key, structs; ret::Bool)
    (key === nothing || !haskey(structs, key)) && return nothing
    v = aapcs64_record_abi(key, structs; ret)
    v.kind === :unknown && return "whose AAPCS64 class cannot be read from DWARF ($(v.reason))"
    return nothing
end

"""
    aapcs64_sret_statement(key, structs) -> String

The statement for the hidden return pointer the thunk passes first for a
pass-by-reference record (`DW_CC_pass_by_reference`): `sret:<bytes>`, the
record's size. `buildCallShape` under AAPCS64 stamps `llvm.sret` of that many
bytes on the pointer, which puts it in x8, the indirect result register; a
plain pointer would go in x0. SysV and Win64 want it in the first argument
register, which a plain pointer already is, so they state nothing.
"""
aapcs64_sret_statement(key, structs) = "sret:$(_sysv_record_size(structs[String(key)]))"

"""
    record_statement(key, structs; ret, abi) -> String

The statement the thunk generator puts on a call for the record `key` crossing
by value, under `abi`: `aapcs64_statement` for `:aapcs64`, `sysv_statement`
otherwise (Win64 reads no statement, and the SysV one is what every x86-64
module always carried). `record_statement_attrs` names the attributes it goes
in.
"""
record_statement(key, structs; ret::Bool, abi::Symbol)::String =
    abi === :aapcs64 ? aapcs64_statement(key, structs; ret) : sysv_statement(key, structs)

"The `(args, ret)` attribute names a call's record statements travel in under `abi`."
record_statement_attrs(abi::Symbol) =
    abi === :aapcs64 ? ("aapcs64_args", "aapcs64_ret") : ("sysv_args", "sysv_ret")
