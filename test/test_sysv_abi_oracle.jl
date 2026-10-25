# =============================================================================
# The SysV struct classification against LLVM's ABI lowering library
#
# FunctionGen classifies a record passed or returned by value from its DWARF
# facts (`sysv_record_abi`, src/IRGen/ir_gen/SysVRecordABI.jl) and states the
# verdict on the call (`sysv_args`, `sysv_ret`); buildCallShape
# (src/mlir/impl/JLCSPasses.cpp) decides the call from it on x86-64 SysV: in
# memory (byval, sret) or in registers, one scalar per eightbyte. LLVM 23 ships
# its own classifier for those rules (llvm/ABI), written to match clang. This
# test asks both about the same records and lists every disagreement.
#
#   corpus  one C unit: the records abi_fuzz generates for SAO_SEEDS (its own
#           generator, test/tools/abi_fuzz/model.jl) with the functions it
#           generates for them, plus the hand-written records of SAO_CURATED.
#           Each record also gets `void take_R(R)` and `R make_R(void)`. Built
#           once as C; the DWARF facts come back in compilation_metadata.json.
#   JLCS    the module `generate_jlcs_ir` writes for that metadata, every
#           jlcs.ffe_call shaped by the real buildCallShape with the SysV
#           target (`jlcs_probe_sysv_call_shapes`), on any host.
#   LLVM    the same records as llvm::abi types, built from the same DWARF
#           facts (offsets, sizes, bitfields, alignment), classified by
#           `createX86_64TargetInfo` (`jlcs_abi_oracle_sysv`,
#           src/mlir/impl/JLCSAbiOracle.cpp). Its reading of the records is
#           itself checked against clang's lowering of the hand-written ones.
#
# Both sides read one description of each record, so a wrong DWARF fact is not
# caught here: this checks classification and call shape. The records have
# this host's layout, which on Windows means MS bitfields and a 4-byte `long`;
# the check is the SysV algorithm on that layout. The library has no Win64
# classifier, so `classifyWin64Struct` is not checked.
#
# A disagreement is reported with its function and position, both answers, the
# record's C definition, and a minimal record: its DWARF members (and those of
# the records inside it) deleted one at a time while the same two answers
# persist. Every disagreement fails. The report names the shape of each
# (SAO_SHAPES), which is what the classification got wrong; until 2026-10-24,
# when the dialect classified from StructGen's emitted body, five of those
# shapes were known defects and `@test_broken`.
#
# Needs libJLCS built against an LLVM whose ABI library has the x86-64
# classifier (LLVM 23+; `:llvm_abi`). Skips elsewhere.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

if requires("SysV struct classification vs LLVM's ABI library", :libJLCS, :llvm_abi)

using RepliBuild.MLIRNative
using RepliBuild.JLCSIRGenerator
using RepliBuild.DWARFParser

isdefined(@__MODULE__, :AbiFuzzModel) ||
    include(joinpath(@__DIR__, "tools", "abi_fuzz", "model.jl"))

const SAO_AF = AbiFuzzModel
const SAO_FG = RepliBuild.JLCSIRGenerator.FunctionGen
const SAO_W = RepliBuild.Wrapper
const SAO_SEEDS = 1:60

# ── Disagreement shapes ───────────────────────────────────────────────────────
#
# What a disagreement looks like, for the report. REG lists the class of each
# eightbyte, MEM is memory with the byval alignment, SRET a hidden return
# pointer with the alignment the callee may assume (see SaoVerdict). Every one
# fails; the shape says which way the classification went wrong.
const SAO_SHAPES = Dict(
    :int_for_sse => "an eightbyte C passes in an SSE register, or in none, goes in a general register",
    :reg_for_mem => "C passes the record in memory, JLCS in registers",
    :mem_for_reg => "JLCS passes in memory a record C passes in registers",
    :underaligned => "C gives the record a more aligned byval slot or sret buffer than JLCS does",
    :x87 => "C returns the record in st0 (X87), JLCS in a register",
    :short_reg => "JLCS passes fewer eightbytes than C does",
    :cascade => "a later crossing of a call in which an earlier record already disagreed: the two " *
                "sides have different registers left",
    :context => "a record both sides agree on alone disagrees inside a call: register accounting " *
                "(buildCallShape's free-register counts) differs",
)

# ── Corpus ────────────────────────────────────────────────────────────────────
#
# Hand-written records: (name, C definition, expected answer as an argument and
# as a return): `:agree`, or `:trap` where FunctionGen refuses the crossing on
# SysV and that is the right answer. Each is the smallest record for one
# feature of the classification, and none depends on the host's layout. In
# dependency order. The comments name what the record exercises; before
# 2026-10-24 the first group, the nested records, the long double, `_Complex`
# and 16-aligned records all disagreed with the oracle, and the float union
# and the two-bitfield record were refused as unclassifiable.
const SAO_CURATED = [
    # an eightbyte holding a float and padding is SSE
    ("c_f_d",      "typedef struct { float x; double y; } c_f_d;", :agree, :agree),
    ("c_d_f",      "typedef struct { double d; float f; } c_d_f;", :agree, :agree),
    ("c_f_pad",    "typedef struct { float x; int : 32; double y; } c_f_pad;", :agree, :agree),
    ("c_p_f",      "typedef struct { void *p; float f; } c_p_f;", :agree, :agree),
    ("c_c_d",      "typedef struct { char c; double d; } c_c_d;", :agree, :agree),
    ("c_f_i",      "typedef struct { float x; int i; } c_f_i;", :agree, :agree),
    ("c_f2",       "typedef struct { float x, y; } c_f2;", :agree, :agree),
    ("c_f3",       "typedef struct { float x, y, z; } c_f3;", :agree, :agree),
    ("c_fa3",      "typedef struct { float v[3]; } c_fa3;", :agree, :agree),
    ("c_d2",       "typedef struct { double a, b; } c_d2;", :agree, :agree),
    ("c_d3",       "typedef struct { double a, b, c; } c_d3;", :agree, :agree),
    ("c_s_f",      "typedef struct { short s; float f; } c_s_f;", :agree, :agree),
    ("c_c3",       "typedef struct { char c[3]; } c_c3;", :agree, :agree),
    ("c_ll_c",     "typedef struct { long long a, b; char c; } c_ll_c;", :agree, :agree),
    ("c_nest_f2d", "typedef struct { c_f2 in; double d; } c_nest_f2d;", :agree, :agree),
    # a nested record that is not tiled, inside one that is
    ("c_wrap_df",  "typedef struct { c_d_f in; } c_wrap_df;", :agree, :agree),
    ("c_pk_sf",    "typedef struct __attribute__((packed)) { char c; c_s_f in; } c_pk_sf;", :agree, :agree),
    # misaligned fields make a record MEMORY, a nested record's own position included
    ("c_pk_ci",    "typedef struct __attribute__((packed)) { char c; int i; } c_pk_ci;", :agree, :agree),
    ("c_pk_cf",    "typedef struct __attribute__((packed)) { char c; float f; } c_pk_cf;", :agree, :agree),
    ("c_pk_ii",    "typedef struct __attribute__((packed)) { int a, b; } c_pk_ii;", :agree, :agree),
    ("c_pp2_cd",   "#pragma pack(push, 2)\ntypedef struct { char c; double d; } c_pp2_cd;\n#pragma pack(pop)", :agree, :agree),
    ("c_al8_c",    "typedef struct __attribute__((aligned(8))) { char x; } c_al8_c;", :agree, :agree),
    ("c_pk_alnest", "#pragma pack(push, 1)\ntypedef struct { char c; c_al8_c s; } c_pk_alnest;\n#pragma pack(pop)", :agree, :agree),
    # unions merge every member
    ("c_u_fi",     "typedef union { float f; int i; } c_u_fi;", :agree, :agree),
    ("c_u_ff",     "typedef union { float f; float g; } c_u_ff;", :agree, :agree),
    ("c_u_ip",     "typedef union { int i; void *p; } c_u_ip;", :agree, :agree),
    ("c_u_pk",     "typedef union { c_pk_ci p; int i; } c_u_pk;", :agree, :agree),
    ("c_u_ld_i",   "typedef union { long double x; int i; } c_u_ld_i;", :agree, :agree),
    # bitfields classify by their bits; an unnamed one is padding
    ("c_bf",       "typedef struct { int a : 3; int b : 5; } c_bf;", :agree, :agree),
    ("c_bf_f",     "typedef struct { int a : 3; float f; } c_bf_f;", :agree, :agree),
    ("c_bf2_f",    "typedef struct { int a : 3; int b : 5; float f; } c_bf2_f;", :agree, :agree),
    ("c_pp1_bf",   "#pragma pack(push, 1)\ntypedef struct { char c; int b : 31; } c_pp1_bf;\n#pragma pack(pop)", :agree, :agree),
    ("c_ubf_f",    "typedef struct { int : 32; float f; } c_ubf_f;", :agree, :agree),
    ("c_ubf_d",    "typedef struct { long long : 64; double d; } c_ubf_d;", :agree, :agree),
    # x87, __int128 and over-alignment. A `long double` record comes back in
    # st0, where no thunk looks: FunctionGen refuses the return.
    ("c_ld",       "typedef struct { long double x; } c_ld;", :agree, :trap),
    ("c_pk_c_ld",  "typedef struct __attribute__((packed)) { char c; long double x; } c_pk_c_ld;", :agree, :agree),
    ("c_i128",     "typedef struct { __int128 x; } c_i128;", :agree, :agree),
    ("c_i128_l",   "typedef struct { __int128 x; long long y; } c_i128_l;", :agree, :agree),
    ("c_al16_f",   "typedef struct { float f __attribute__((aligned(16))); } c_al16_f;", :agree, :agree),
    ("c_al16_big", "typedef struct __attribute__((aligned(16))) { long long a, b, c; } c_al16_big;", :agree, :agree),
    # _Complex
    ("c_cf",       "typedef struct { _Complex float z; } c_cf;", :agree, :agree),
    ("c_cd",       "typedef struct { _Complex double z; } c_cd;", :agree, :agree),
    ("c_f_cf",     "typedef struct { float a; _Complex float z; } c_f_cf;", :agree, :agree),
    # an empty record takes no register (GNU C)
    ("c_empty",    "typedef struct { } c_empty;", :agree, :agree),
]

"`u` with every ordinary identifier prefixed by `p`, so many units share one C file."
function sao_prefixed(u, p::String)
    v = deepcopy(u)
    for r in v.records
        isempty(r.name) || (r.name = p * r.name)
        isempty(r.tag) || (r.tag = p * r.tag)
    end
    for e in v.enums
        e.name = p * e.name
        isempty(e.tag) || (e.tag = p * e.tag)
        e.names = [p * n for n in e.names]
    end
    for f in v.funcs
        f.name = p * f.name
    end
    return v
end

sao_twin(u, t) = t isa SAO_AF.EnumT && u.enums[t.e].twin != 0
sao_zeroed(spell) = "{ $spell r; memset(&r, 0, sizeof r); return r; }"

"""
    sao_corpus() -> (source, records, cdefs)

The corpus C text, every record's name in the order defined, and each record's
C definition by name.
"""
function sao_corpus()
    io = IOBuffer()
    println(io, "/* SysV ABI oracle corpus, generated by test/test_sysv_abi_oracle.jl */")
    println(io, "#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n#define AF_BOOL _Bool\n")
    records = Tuple{String,String}[]          # (record name, C spelling)
    cdefs = Dict{String,String}()
    for (name, def, _, _) in SAO_CURATED
        println(io, def)
        push!(records, (name, name))
        cdefs[name] = def
    end
    stubs = IOBuffer()
    for seed in SAO_SEEDS
        u = sao_prefixed(SAO_AF.generate(seed), "s$(seed)_")
        println(io, "\n/* abi_fuzz seed $seed */")
        for e in u.enums
            e.twin == 0 && SAO_AF.emit_enum(io, u, e)
        end
        for r in SAO_AF.named_records(u)
            def = sprint(SAO_AF.emit_record, u, r)
            print(io, def)
            push!(records, (u.records[r].name, SAO_AF.rec_spell(u, r)))
            cdefs[u.records[r].name] = strip(def)
        end
        # The unit's own functions, as stubs: where they pass what matters,
        # not what they do.
        for f in u.funcs
            (sao_twin(u, f.ret) || any(p -> sao_twin(u, p.ty), f.params)) && continue
            rs = f.ret === nothing ? "void" : SAO_AF.type_spell(u, f.ret)
            ps = isempty(f.params) ? "void" : join((SAO_AF.isptrparam(p.ty) ?
                "const $(SAO_AF.rec_spell(u, ((p.ty::SAO_AF.ArrT).elem::SAO_AF.RecT).r)) *$(p.name)" :
                "$(SAO_AF.type_spell(u, p.ty)) $(p.name)" for p in f.params), ", ")
            println(stubs, rs, " ", f.name, "(", ps, ") ", f.ret === nothing ? "{ }" : sao_zeroed(rs))
        end
    end
    println(io, "\n/* every record, each way */")
    for (name, spell) in records
        println(io, "void take_$(name)($(spell) x) { (void)x; }")
        println(io, "$(spell) make_$(name)(void) ", sao_zeroed(spell))
    end
    println(io, "\n/* the abi_fuzz units' functions */")
    print(io, String(take!(stubs)))
    return String(take!(io)), first.(records), cdefs
end

"Build the corpus as C in `dir`; the metadata it wrote."
function sao_build(dir::String, src::String)
    mkpath(joinpath(dir, "src"))
    write(joinpath(dir, "src", "corpus.c"), src)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "sysv_oracle_corpus"
    root = "$(escape_string(dir))"

    [compile]
    flags = ["-O1", "-fPIC"]
    source_files = ["$(escape_string(joinpath(dir, "src", "corpus.c")))"]

    [link]
    enable_lto = false

    [binary]
    type = "shared"

    [wrap]
    language = "c"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    RepliBuild.build(toml)
    return RepliBuild._read_json(joinpath(dir, "julia", "compilation_metadata.json"))
end

# ── The LLVM side: DWARF facts as llvm::abi types ────────────────────────────

const SAO_INT_BITS = Dict(
    "char" => 8, "signed char" => 8, "unsigned char" => 8, "_Bool" => 8, "bool" => 8,
    "short" => 16, "unsigned short" => 16, "short int" => 16, "short unsigned int" => 16,
    "int" => 32, "unsigned int" => 32, "long long" => 64, "unsigned long long" => 64,
    "long long int" => 64, "long long unsigned int" => 64,
    "int8_t" => 8, "uint8_t" => 8, "int16_t" => 16, "uint16_t" => 16,
    "int32_t" => 32, "uint32_t" => 32, "int64_t" => 64, "uint64_t" => 64,
    "size_t" => 64, "ssize_t" => 64, "intptr_t" => 64, "uintptr_t" => 64, "ptrdiff_t" => 64,
    "__int128" => 128, "unsigned __int128" => 128, "__int128_t" => 128, "__uint128_t" => 128)

sao_int(bits; signed = true) = Dict{String,Any}("k" => "int", "bits" => bits,
                                                "align" => clamp(bits ÷ 8, 1, 16), "signed" => signed)
sao_float(sem, align) = Dict{String,Any}("k" => "float", "sem" => sem, "align" => align)
sao_bytes(x) = SAO_W._parse_int_or_hex(something(x, 0))

"""
    sao_type(c_type, size, meta) -> Dict

One C type as the oracle reads it (see JLCSAbiOracle.cpp). `size` is the bytes
the metadata gives the value, which settles `long` and the typedef'd integers.
Throws on a type it cannot describe, which is a corpus this test does not know.
"""
function sao_type(c_type::AbstractString, size::Integer, meta, seen::Set{String} = Set{String}())
    sd = meta["struct_definitions"]
    t = String(strip(replace(replace(c_type, r"\b(?:const|volatile)\b" => ""), r"\s+" => " ")))
    (endswith(t, "*") || endswith(t, "&") || occursin("(*)", t)) && return Dict{String,Any}("k" => "ptr")
    am = match(r"^(.*?)\s*\[(\d*)\]((?:\[\d*\])*)$", t)
    if am !== nothing
        n = isempty(am.captures[2]) ? 0 : parse(Int, am.captures[2])
        elem = String(am.captures[1]) * String(am.captures[3])
        return Dict{String,Any}("k" => "array", "n" => n, "size_bits" => 8 * size,
                                "elem" => sao_type(elem, n > 0 ? size ÷ n : 0, meta, seen))
    end
    t == "float" && return sao_float("single", 4)
    t == "double" && return sao_float("double", 8)
    t == "long double" && return sao_float("x87", 16)
    t in ("_Float16", "__fp16") && return sao_float("half", 2)
    cm = match(r"^(?:_Complex|complex) (float|double|long double)$", t)
    if cm !== nothing
        elem = sao_type(cm.captures[1], 0, meta, seen)
        return Dict{String,Any}("k" => "complex", "elem" => elem, "align" => elem["align"])
    end
    if haskey(SAO_INT_BITS, t)
        bits = t in ("char", "_Bool", "bool") || size <= 0 ? SAO_INT_BITS[t] : 8 * size
        return sao_int(bits; signed = !(startswith(t, "unsigned") || startswith(t, "uint") ||
                                        t in ("_Bool", "bool")))
    end
    t in ("long", "long int", "unsigned long", "long unsigned int") &&
        return sao_int(8 * (size > 0 ? size : sizeof(Clong)); signed = !occursin("unsigned", t))
    t = String(replace(t, r"^(?:struct|union|enum)\s+" => ""))
    haskey(sd, "__enum__" * t) && return sao_int(8 * SAO_W._parse_dwarf_size(sd["__enum__"*t]))
    haskey(sd, t) && return sao_record(t, meta, seen)
    facts = get(get(meta, "record_abi", Dict()), t, nothing)
    if facts !== nothing && get(facts, "empty", false) == true
        return Dict{String,Any}("k" => "record", "size_bits" => 8 * sao_bytes(get(facts, "byte_size", 0)),
                                "align" => 1, "union" => false, "fields" => Any[])
    end
    error("the oracle cannot describe C type `$c_type` (size $size)")
end

"The alignment of an oracle type."
sao_align(ty) = ty["k"] == "ptr" ? 8 : ty["k"] == "array" ? sao_align(ty["elem"]) : Int(ty["align"])

"A metadata record as an oracle record."
function sao_record(key::String, meta, seen::Set{String} = Set{String}())
    sd = meta["struct_definitions"]
    key in seen && error("record `$key` contains itself by value")
    push!(seen, key)
    info = sd[key]
    is_union = get(info, "kind", "struct") == "union"
    fields = Any[]
    for m in get(info, "members", [])
        ty = sao_type(String(get(m, "c_type", "")), sao_bytes(get(m, "size", 0)), meta, seen)
        if haskey(m, "bit_size")
            haskey(m, "data_bit_offset") || is_union ||
                error("bitfield `$(get(m, "name", "?"))` of `$key` has no data_bit_offset")
            push!(fields, Dict{String,Any}("offset_bits" => Int(get(m, "data_bit_offset", 0)),
                                           "type" => ty, "bit_width" => Int(m["bit_size"])))
        else
            off = is_union ? 0 : sao_bytes(get(m, "offset", 0))
            push!(fields, Dict{String,Any}("offset_bits" => 8 * off, "type" => ty))
        end
    end
    delete!(seen, key)
    size = SAO_W._parse_dwarf_size(info)
    # The record's alignment, exactly where DWARF has it. Clang writes
    # DW_AT_alignment as the record's actual alignment whenever `aligned` or
    # `#pragma pack` changed it (`aligned(4)` on a record of doubles: 8;
    # `pack(2)`: 2), and leaves it out for a plain record and for
    # `__attribute__((packed))`. `_c_record_align`, a bound for sret buffers,
    # answers the larger of that and the members' alignment, so it is used
    # only without the attribute, and then lowered to the largest alignment
    # the layout is consistent with: a packed record's is 1, not its members'.
    if haskey(info, "alignment")
        al = SAO_W._c_explicit_align(info["alignment"])
    else
        al = SAO_W._c_record_align(info, sd)
        fits(a) = size % a == 0 && all(f -> haskey(f, "bit_width") ||
                                       f["offset_bits"] % (8 * min(a, sao_align(f["type"]))) == 0, fields)
        while al > 1 && !fits(al)
            al ÷= 2
        end
    end
    return Dict{String,Any}("k" => "record", "size_bits" => 8 * size, "align" => al,
                            "union" => is_union, "fields" => fields)
end

sao_ret_ctype(f) = (rt = get(f, "return_type", nothing); rt === nothing ? "void" : String(get(rt, "c_type", "void")))

function sao_signature(f, meta)
    rct = sao_ret_ctype(f)
    ret = rct == "void" ? Dict{String,Any}("k" => "void") :
          sao_type(rct, sao_bytes(get(get(f, "return_type", Dict()), "size", 0)), meta)
    params = [sao_type(String(get(p, "c_type", "")), sao_bytes(get(p, "size", 0)), meta)
              for p in get(f, "parameters", [])]
    return Dict("name" => String(f["name"]), "ret" => ret, "params" => params)
end

"Ask the oracle about `fns`: name => its answer."
function sao_oracle(fns, meta)
    req = Dict("functions" => [sao_signature(f, meta) for f in fns])
    text = unsafe_string(ccall((:jlcs_abi_oracle_sysv, MLIRNative.libJLCS), Cstring, (Cstring,),
                               RepliBuild.JSON.json(req)))
    ans = RepliBuild.JSON.parse(text)
    haskey(ans, "error") && error("the oracle refused the request: $(ans["error"])")
    return Dict(String(a["name"]) => a for a in ans["functions"])
end

# ── The JLCS side: the generator's module, shaped by buildCallShape ──────────

const SAO_EMPTY_VT = DWARFParser.VtableInfo(Dict{String,DWARFParser.ClassInfo}(),
                                            Dict{String,UInt64}(), Dict{String,UInt64}())

"""
    sao_jlcs(meta) -> (shapes, traps)

`shapes`: callee => fields (the return, then one per operand) from
`jlcs_probe_sysv_call_shapes` over the module `generate_jlcs_ir` writes for
`meta`. `traps`: callee => why FunctionGen gave it no thunk.
"""
function sao_jlcs(meta)
    ir = JLCSIRGenerator.generate_jlcs_ir(SAO_EMPTY_VT, meta)
    traps = Dict(String(m.captures[1]) => String(m.captures[2])
                 for m in eachmatch(r"(?m)^// no thunk for (\S+): (.*)$", ir))
    ctx = create_context()
    try
        mod = parse_module(ctx, ir)
        text = unsafe_string(ccall((:jlcs_probe_sysv_call_shapes, MLIRNative.libJLCS), Cstring,
                                   (MLIRNative.MlirModule,), mod))
        shapes = Dict{String,Vector{String}}()
        for line in split(text, '\n'; keepempty = false)
            parts = String.(split(line, '\t'))
            shapes[parts[1]] = parts[2:end]
        end
        return shapes, traps
    finally
        destroy_context(ctx)
    end
end

# ── Verdicts ──────────────────────────────────────────────────────────────────
#
# Both answers reduced to where the record goes. REG lists the class of each
# eightbyte (NONE: a low eightbyte of padding C passes nowhere), MEM is memory
# with the byval alignment, SRET a hidden return pointer with the alignment the
# callee may assume, IGNORE nothing at all. An argument the oracle passes
# directly on the stack because no register was left is MEM 8: the stack slot
# a byval of alignment 8 gets.

struct SaoVerdict
    kind::Symbol                 # :REG, :MEM, :SRET, :IGNORE, :ODD
    classes::Vector{String}
    align::Int
    raw::String
end

Base.show(io::IO, v::SaoVerdict) = print(io,
    v.kind === :REG ? "REG[" * join(v.classes, ",") * "]" :
    v.kind === :MEM ? "MEM(align $(v.align))" :
    v.kind === :SRET ? "SRET(align $(v.align))" :
    v.kind === :IGNORE ? "IGNORE" : "?($(v.raw))")

# `none` is a low eightbyte of padding, which the probe prints before the one
# scalar it passes for the high eightbyte.
const SAO_JLCS_CLASS = Dict("i64" => "INTEGER", "f64" => "SSE", "none" => "NONE")
const SAO_DROPPED = SaoVerdict(:IGNORE, String[], 0, "no operand")

function sao_jlcs_verdict(field::AbstractString)
    if startswith(field, "regs:")
        return SaoVerdict(:REG, [get(SAO_JLCS_CLASS, t, "?" * t) for t in split(field[6:end], ',')], 0, field)
    elseif startswith(field, "byval:")
        return SaoVerdict(:MEM, String[], parse(Int, field[7:end]), field)
    elseif startswith(field, "sret:")
        return SaoVerdict(:SRET, String[], parse(Int, field[6:end]), field)
    elseif field == "void" || field == "ignore"
        return SaoVerdict(:IGNORE, String[], 0, field)
    end
    return SaoVerdict(:ODD, String[], 0, field)
end

function sao_oracle_verdict(a; ret::Bool)
    k = a["kind"]
    raw = k in ("direct", "extend") ? "coerce " * String(a["coerce"]) : RepliBuild.JSON.json(a)
    k == "ignore" && return SaoVerdict(:IGNORE, String[], 0, raw)
    k == "indirect" && return SaoVerdict(ret ? :SRET : :MEM, String[], Int(a["align"]), raw)
    if k in ("direct", "extend")
        (ret || get(a, "in_registers", true)) && return SaoVerdict(:REG, String.(a["classes"]), 0, raw)
        return SaoVerdict(:MEM, String[], 8, raw * ", on the stack")
    end
    return SaoVerdict(:ODD, String[], 0, raw)
end

function sao_agree(j::SaoVerdict, o::SaoVerdict)
    j.kind === o.kind || return false
    j.kind === :REG && return j.classes == o.classes
    j.kind === :MEM && return j.align == o.align
    # a caller-owned return slot may be more aligned than the callee assumes
    j.kind === :SRET && return j.align >= o.align
    return j.kind === :IGNORE
end

"The shape of a disagreement (a key of SAO_SHAPES), or `nothing` when none fits."
function sao_family(j::SaoVerdict, o::SaoVerdict)
    if j.kind === :REG && o.kind === :REG
        "X87" in o.classes && return :x87
        length(j.classes) < length(o.classes) && return :short_reg
        ok = all(eachindex(j.classes)) do i
            jc, oc = j.classes[i], get(o.classes, i, "NONE")
            jc == oc || (jc == "INTEGER" && oc in ("SSE", "NONE"))
        end
        return ok ? :int_for_sse : nothing
    end
    j.kind === :REG && o.kind in (:MEM, :SRET) && return :reg_for_mem
    j.kind in (:MEM, :SRET) && o.kind === :REG && return :mem_for_reg
    j.kind === o.kind && j.kind in (:MEM, :SRET) && j.align < o.align && return :underaligned
    return nothing
end

# ── Comparison ────────────────────────────────────────────────────────────────

struct SaoCase
    fn::String
    pos::String                  # "ret" or "arg <k>"
    record::String               # the record's metadata key
    jlcs::SaoVerdict
    llvm::SaoVerdict
    family::Union{Symbol,Nothing} # nothing when they agree, or when no family fits
end

sao_crossings(f, sd, rabi) =
    (SAO_FG.by_value_crossing(sao_ret_ctype(f), sd, rabi),
     [SAO_FG.by_value_crossing(String(get(p, "c_type", "")), sd, rabi) for p in get(f, "parameters", [])])

"""
    sao_sysv_trapped(f, meta) -> Bool

Whether FunctionGen refuses `f` on a SysV host. `thunk_crossings` asks
`sysv_trap_reason` for the host's convention, so on Windows it builds thunks a
SysV host refuses (a `long double` record returned in st0, a record whose
class DWARF does not settle). Asked here for SysV, the one answer this test is
about.
"""
function sao_sysv_trapped(f, meta)
    sd = meta["struct_definitions"]
    (ret, params) = sao_crossings(f, sd, get(meta, "record_abi", Dict{String,Any}()))
    ret[1] === :layout && SAO_FG.sysv_trap_reason(ret[2], sd; ret = true, abi = :sysv) !== nothing && return true
    return any(((k, key),) -> k === :layout && SAO_FG.sysv_trap_reason(key, sd; ret = false, abi = :sysv) !== nothing,
               params)
end

"""
    sao_compare(fns, meta, shapes, oracle) -> (cases, gaps)

One `SaoCase` per by-value record crossing of every function in `fns` with a
SysV thunk. `gaps` names what could not be compared, and why.
"""
function sao_compare(fns, meta, shapes, oracle)
    sd = meta["struct_definitions"]
    rabi = get(meta, "record_abi", Dict{String,Any}())
    cases = SaoCase[]
    gaps = String[]
    for f in fns
        name = String(f["name"])
        fields = get(shapes, name, nothing)
        (fields === nothing || sao_sysv_trapped(f, meta)) && continue
        a = oracle[name]
        if haskey(a, "error")
            push!(gaps, "$name: the oracle could not read it: $(a["error"])")
            continue
        end
        if any(x -> startswith(x, "unmodelled") || startswith(x, "inconsistent") || startswith(x, "error"), fields)
            push!(gaps, "$name: the probe could not shape it: $(join(fields, " "))")
            continue
        end
        (ret, params) = sao_crossings(f, sd, rabi)
        mine = SaoCase[]
        add(pos, key, j, o) = begin
            fam = sao_agree(j, o) ? nothing :
                  any(c -> c.family !== nothing || !sao_agree(c.jlcs, c.llvm), mine) ? :cascade :
                  sao_family(j, o)
            push!(mine, SaoCase(name, pos, key, j, o, fam))
        end
        if ret[1] in (:layout, :ignore)
            add("ret", something(ret[2], sao_ret_ctype(f)),
                ret[1] === :ignore ? SAO_DROPPED : sao_jlcs_verdict(fields[1]),
                sao_oracle_verdict(a["ret"]; ret = true))
        end
        # FunctionGen passes no operand for a dropped (empty) record, which has
        # no table entry and is named by its C spelling.
        operand = 1
        for (k, (pk, pkey)) in enumerate(params)
            if pk === :ignore
                add("arg $k", String(get(f["parameters"][k], "c_type", "")), SAO_DROPPED,
                    sao_oracle_verdict(a["params"][k]; ret = false))
                continue
            end
            operand += 1
            if operand > length(fields)
                push!(gaps, "$name: the probe shaped $(length(fields) - 1) operands for $(length(params)) parameters")
                break
            end
            pk === :layout || continue
            add("arg $k", pkey, sao_jlcs_verdict(fields[operand]),
                sao_oracle_verdict(a["params"][k]; ret = false))
        end
        operand == length(fields) || any(g -> startswith(g, name * ":"), gaps) ||
            push!(gaps, "$name: the probe shaped $(length(fields) - 1) operands, $(operand - 1) were matched")
        append!(cases, mine)
    end
    # A disagreement inside a call whose record agrees alone (take_R, make_R)
    # is the call's doing, unless an earlier crossing already disagreed.
    alone = Dict((c.record, c.pos == "ret") => sao_agree(c.jlcs, c.llvm)
                 for c in cases if occursin(r"^(?:take|make)_", c.fn))
    for (i, c) in enumerate(cases)
        (c.family === nothing || c.family === :cascade || occursin(r"^(?:take|make)_", c.fn)) && continue
        get(alone, (c.record, c.pos == "ret"), false) &&
            (cases[i] = SaoCase(c.fn, c.pos, c.record, c.jlcs, c.llvm, :context))
    end
    return cases, gaps
end

# ── The oracle against clang ─────────────────────────────────────────────────
#
# Whether the oracle reads the records right: the DWARF facts it is given, as
# llvm::abi types, should get the answers clang's own lowering gives the C
# definitions. Checked on the hand-written records, whose layout is the same on
# every host, from the IR clang emits for x86_64-unknown-linux-gnu.

"Split `s` on commas outside brackets and parentheses."
function sao_split(s::AbstractString)
    parts = String[]
    depth = 0
    start = firstindex(s)
    for i in eachindex(s)
        c = s[i]
        if c in ('(', '{', '<', '[')
            depth += 1
        elseif c in (')', '}', '>', ']')
            depth -= 1
        elseif c == ',' && depth == 0
            push!(parts, strip(s[start:prevind(s, i)]))
            start = nextind(s, i)
        end
    end
    tail = strip(s[start:end])
    isempty(tail) || push!(parts, tail)
    return parts
end

"The register class of each eightbyte an IR type clang passes directly takes."
function sao_ir_classes(t::AbstractString)
    t = strip(t)
    startswith(t, "{") && return reduce(vcat, sao_ir_classes.(sao_split(t[2:end-1])); init = String[])
    t == "x86_fp80" && return ["X87"]
    t == "i128" && return ["INTEGER", "INTEGER"]
    (t == "ptr" || occursin(r"^i\d+$", t)) && return ["INTEGER"]
    (t in ("half", "bfloat", "float", "double", "fp128") || startswith(t, "<")) && return ["SSE"]
    return ["?" * t]
end

# The leading type of an IR parameter: `{ ... }`, `< ... >` or one word.
function sao_ir_type(p::AbstractString)
    p = strip(p)
    for (open, close) in (('{', '}'), ('<', '>'))
        if startswith(p, open)
            depth = 0
            for i in eachindex(p)
                depth += p[i] == open ? 1 : p[i] == close ? -1 : 0
                depth == 0 && return p[1:i]
            end
        end
    end
    return first(split(p))
end

"""
    sao_clang_verdicts(defs, names) -> Dict(fn => SaoVerdict)

What clang's x86_64-unknown-linux-gnu lowering does with `take_R` and `make_R`
for each record `R` in `names`, defined by the C text `defs`.
"""
function sao_clang_verdicts(defs::String, names)
    dir = mktempdir()
    src = joinpath(dir, "curated.c")
    ll = joinpath(dir, "curated.ll")
    write(src, "void *memset(void *, int, unsigned long);\n", defs, "\n",
          join(("void take_$n($n x) { (void)x; }\n$n make_$n(void) $(sao_zeroed(n))" for n in names), "\n"))
    run(`$(Sys.which("clang")) --target=x86_64-unknown-linux-gnu -O1 -S -emit-llvm -w -o $ll $src`)
    out = Dict{String,SaoVerdict}()
    for line in eachline(ll)
        startswith(line, "define ") || continue
        m = match(r"@((?:take|make)_\w+)\(", line)
        m === nothing && continue
        fn = String(m.captures[1])
        head = line[1:m.offset-2]
        # the parameters: up to the parenthesis that closes the list
        depth = 0
        stop = lastindex(line)
        for i in (m.offset + length(m.match) - 1):lastindex(line)
            depth += line[i] == '(' ? 1 : line[i] == ')' ? -1 : 0
            depth == 0 && (stop = i; break)
        end
        params = sao_split(line[m.offset+length(m.match):prevind(line, stop)])
        sret = findfirst(p -> occursin("sret(", p), params)
        if startswith(fn, "make_")
            ret = endswith(head, "}") ? head[findlast('{', head):end] :
                  endswith(head, ">") ? head[findlast('<', head):end] : String(last(split(head)))
            out[fn] = sret !== nothing ?
                SaoVerdict(:SRET, String[], parse(Int, match(r"\balign (\d+)", params[sret]).captures[1]), line) :
                ret == "void" ? SaoVerdict(:IGNORE, String[], 0, line) :
                SaoVerdict(:REG, sao_ir_classes(ret), 0, line)
        else
            byval = findfirst(p -> occursin("byval(", p), params)
            out[fn] = byval !== nothing ?
                SaoVerdict(:MEM, String[], parse(Int, match(r"\balign (\d+)", params[byval]).captures[1]), line) :
                isempty(params) ? SaoVerdict(:IGNORE, String[], 0, line) :
                SaoVerdict(:REG, reduce(vcat, (sao_ir_classes(sao_ir_type(p)) for p in params)), 0, line)
        end
    end
    return out
end

# NONE (a padding eightbyte passed nowhere) and SSEUP (the rest of one XMM
# register) do not show in IR, which lists only the values passed.
sao_ir_view(v::SaoVerdict) = v.kind === :REG ?
    SaoVerdict(:REG, filter(c -> !(c in ("NONE", "SSEUP")), v.classes), 0, v.raw) : v

# ── Minimal records ──────────────────────────────────────────────────────────

"The metadata keys `key` holds by value, itself included."
function sao_closure(key::String, sd, out::Set{String} = Set{String}())
    key in out && return out
    push!(out, key)
    for m in get(sd[key], "members", [])
        t = String(strip(replace(replace(String(get(m, "c_type", "")), r"\b(?:const|volatile)\b" => ""),
                                 r"(?:\[\d*\])+$" => "")))
        t = String(replace(t, r"^(?:struct|union|enum)\s+" => ""))
        endswith(t, "*") && continue
        haskey(sd, "__enum__" * t) && push!(out, "__enum__" * t)
        haskey(sd, t) && sao_closure(t, sd, out)
    end
    return out
end

"Both answers for `key` alone, as an argument or a return, with `sd` as the records."
function sao_alone(key::String, which::Symbol, sd, meta)
    fn = Dict{String,Any}("name" => "probe", "mangled" => "probe", "is_method" => false,
        "is_vararg" => false,
        "parameters" => which === :arg ? Any[Dict{String,Any}("name" => "x", "c_type" => key)] : Any[],
        "return_type" => Dict{String,Any}("c_type" => which === :ret ? key : "void"))
    m = Dict{String,Any}("language" => "c", "struct_definitions" => sd,
                         "record_abi" => get(meta, "record_abi", Dict{String,Any}()), "functions" => Any[fn])
    shapes, _ = sao_jlcs(m)
    fields = get(shapes, "probe", nothing)
    fields === nothing && return nothing
    a = sao_oracle(m["functions"], m)["probe"]
    haskey(a, "error") && return nothing
    return which === :arg ?
        (sao_jlcs_verdict(fields[2]), sao_oracle_verdict(a["params"][1]; ret = false)) :
        (sao_jlcs_verdict(fields[1]), sao_oracle_verdict(a["ret"]; ret = true))
end

"""
    sao_minimal(key, which, meta) -> Dict

The records `key` holds by value, with members deleted one at a time while
`key` alone still gets the same two answers.
"""
function sao_minimal(key::String, which::Symbol, meta)
    full = meta["struct_definitions"]
    sd = Dict{String,Any}(k => deepcopy(full[k]) for k in sao_closure(key, full))
    want = sao_alone(key, which, sd, meta)
    want === nothing && return sd
    want = string.(want)
    progress = true
    while progress
        progress = false
        for k in sort!(collect(keys(sd))), i in length(get(sd[k], "members", [])):-1:1
            length(sd[k]["members"]) <= 1 && continue
            trial = deepcopy(sd)
            deleteat!(trial[k]["members"], i)
            got = sao_alone(key, which, trial, meta)
            if got !== nothing && string.(got) == want
                sd = trial
                progress = true
                break
            end
        end
    end
    return sd
end

function sao_show_record(io::IO, key::String, sd; indent = "    ")
    info = sd[key]
    println(io, indent, get(info, "kind", "struct"), " ", key, " (", SAO_W._parse_dwarf_size(info),
            " bytes, align ", SAO_W._c_record_align(info, sd), ")")
    for m in get(info, "members", [])
        at = haskey(m, "bit_size") ? "bits $(get(m, "data_bit_offset", 0))+$(m["bit_size"])" :
                                     "@$(sao_bytes(get(m, "offset", 0)))"
        println(io, indent, "    ", rpad(string(get(m, "c_type", "?"), " ", get(m, "name", "?")), 34), at)
    end
end

# ── Running it ────────────────────────────────────────────────────────────────

const SAO_DIR = mktempdir()
const SAO_SRC, SAO_RECORDS, SAO_CDEFS = sao_corpus()
const SAO_META = sao_build(SAO_DIR, SAO_SRC)
const SAO_FNS = [f for f in SAO_META["functions"]
                 if !get(f, "is_vararg", false) && !isempty(String(get(f, "name", "")))]
const SAO_SHAPES, SAO_TRAPS = sao_jlcs(SAO_META)
const SAO_ORACLE = sao_oracle(SAO_FNS, SAO_META)
const SAO_CASES, SAO_GAPS = sao_compare(SAO_FNS, SAO_META, SAO_SHAPES, SAO_ORACLE)

# The record behind a take_/make_ function, by C name.
sao_subject(fn) = (m = match(r"^(?:take|make)_(.+)$", fn); m === nothing ? nothing : String(m.captures[1]))
sao_fn(name) = (i = findfirst(f -> f["name"] == name, SAO_FNS); i === nothing ? nothing : SAO_FNS[i])

"""
    sao_report(io)

Every disagreement, grouped by family: the record's C definition, both answers
for each crossing, and the minimal record for its first one.
"""
function sao_report(io::IO)
    bad = [c for c in SAO_CASES if !sao_agree(c.jlcs, c.llvm)]
    ntrap = count(f -> haskey(SAO_TRAPS, String(f["name"])) || sao_sysv_trapped(f, SAO_META), SAO_FNS)
    println(io, "SysV classification, JLCS vs LLVM's ABI library: $(length(SAO_RECORDS)) records, ",
            "$(length(SAO_FNS)) functions ($ntrap refused on SysV), $(length(SAO_CASES)) record crossings ",
            "compared, $(length(bad)) disagree.")
    groups = Dict{Union{Symbol,Nothing},Vector{SaoCase}}()
    for c in bad
        push!(get!(groups, c.family, SaoCase[]), c)
    end
    for fam in sort!(collect(keys(groups)); by = string)
        cs = groups[fam]
        println(io, "\n[$(something(fam, "UNKNOWN"))] ", get(SAO_SHAPES, fam, "no known shape"))
        byrec = Dict{String,Vector{SaoCase}}()
        for c in cs
            push!(get!(byrec, something(sao_subject(c.fn), c.fn), SaoCase[]), c)
        end
        for subj in sort!(collect(keys(byrec)))
            println(io, "  ", subj)
            def = get(SAO_CDEFS, subj, nothing)
            def === nothing || println(io, "    ", replace(def, '\n' => "\n    "))
            for c in byrec[subj]
                println(io, "    $(c.fn) $(c.pos) [$(c.record)]: JLCS $(c.jlcs) ($(c.jlcs.raw)); LLVM $(c.llvm) ($(c.llvm.raw))")
            end
            c = byrec[subj][1]
            if sao_subject(c.fn) !== nothing && fam !== :cascade
                which = c.pos == "ret" ? :ret : :arg
                sd = sao_minimal(c.record, which, SAO_META)
                println(io, "    minimal record ($(which == :ret ? "returned" : "passed")):")
                for k in sort!(collect(keys(sd)); by = k -> (k != c.record, k))
                    startswith(k, "__enum__") || sao_show_record(io, k, sd; indent = "      ")
                end
            end
        end
    end
end

@testset "SysV struct classification vs LLVM's ABI library" begin
    # Its own testset, so that an error while minimising is reported there and
    # the assertions below still run.
    @testset "report" begin
        report = sprint(sao_report)
        @info report
        @test occursin("$(count(c -> !sao_agree(c.jlcs, c.llvm), SAO_CASES)) disagree", report)
    end

    @testset "harness" begin
        @test isempty(SAO_GAPS)
        # Every function FunctionGen gave a thunk was shaped by the probe, and
        # the oracle answered for every function.
        thunked = sort!([String(f["name"]) for f in SAO_FNS if !haskey(SAO_TRAPS, String(f["name"]))])
        @test thunked == sort!(collect(keys(SAO_SHAPES)))
        @test length(SAO_ORACLE) == length(SAO_FNS)
        # Every record crossed both ways, unless SysV refuses it.
        crossed = Set((c.fn, c.pos == "ret") for c in SAO_CASES)
        checked = 0
        for name in SAO_RECORDS
            for (fn, isret) in (("take_$name", false), ("make_$name", true))
                f = sao_fn(fn)
                @test (fn, f !== nothing && ((fn, isret) in crossed || sao_sysv_trapped(f, SAO_META))) == (fn, true)
                checked += 1
            end
        end
        @test checked == 2 * length(SAO_RECORDS)
        # Every by-value record crossing of every function a SysV host gives a
        # thunk was compared, counted again from the metadata.
        sd, rabi = SAO_META["struct_definitions"], get(SAO_META, "record_abi", Dict{String,Any}())
        expected = 0
        for f in SAO_FNS
            (haskey(SAO_TRAPS, String(f["name"])) || sao_sysv_trapped(f, SAO_META)) && continue
            (ret, params) = sao_crossings(f, sd, rabi)
            expected += count(((k, _),) -> k in (:layout, :ignore), vcat([ret], params))
        end
        @test length(SAO_CASES) == expected
        # The corpus reaches every kind of answer, on both sides.
        agreed = [c.llvm for c in SAO_CASES if sao_agree(c.jlcs, c.llvm)]
        @test any(v -> v.kind === :REG && "SSE" in v.classes, agreed)
        @test any(v -> v.kind === :REG && "INTEGER" in v.classes && "SSE" in v.classes, agreed)
        @test any(v -> v.kind === :MEM, agreed)
        @test any(v -> v.kind === :SRET, agreed)
        @test any(v -> v.kind === :IGNORE, agreed)
        # The register-exhaustion path (`buildCallShape`'s "does not fit"):
        # records that go in registers alone and in memory inside a call,
        # where both sides agree.
        alone = Dict(c.record => c.llvm for c in SAO_CASES if startswith(c.fn, "take_"))
        spilled = [c for c in SAO_CASES if sao_subject(c.fn) === nothing && c.llvm.kind === :MEM &&
                   get(alone, c.record, c.llvm).kind === :REG && sao_agree(c.jlcs, c.llvm)]
        @test length(spilled) >= 5
        @test any(c -> occursin("on the stack", c.llvm.raw), spilled)
    end

    if requires("the oracle against clang's SysV lowering", :clang)
        @testset "the oracle agrees with clang (x86_64-unknown-linux-gnu)" begin
            names = [n for (n, _, _, _) in SAO_CURATED]
            clang = sao_clang_verdicts(join((d for (_, d, _, _) in SAO_CURATED), "\n"), names)
            checked = 0
            for n in names, (fn, isret) in (("take_$n", false), ("make_$n", true))
                a = SAO_ORACLE[fn]
                o = sao_ir_view(isret ? sao_oracle_verdict(a["ret"]; ret = true) :
                                        sao_oracle_verdict(a["params"][1]; ret = false))
                c = get(clang, fn, SaoVerdict(:ODD, String[], 0, "clang emitted no $fn"))
                @test (fn, string(o)) == (fn, string(c))
                checked += 1
            end
            @test checked == 2 * length(names)
        end
    end

    @testset "hand-written records" begin
        checked = 0
        for (name, _, want_arg, want_ret) in SAO_CURATED
            for (fn, want) in (("take_$name", want_arg), ("make_$name", want_ret))
                f = sao_fn(fn)
                cs = [c for c in SAO_CASES if c.fn == fn]
                got = f === nothing ? :missing :
                      sao_sysv_trapped(f, SAO_META) || haskey(SAO_TRAPS, fn) ? :trap :
                      length(cs) != 1 ? :missing :
                      sao_agree(cs[1].jlcs, cs[1].llvm) ? :agree : something(cs[1].family, :unknown)
                @test (fn, got) == (fn, want)
                checked += 1
            end
        end
        @test checked == 2 * length(SAO_CURATED)
        # The refusals are the ones SAO_CURATED says, and no other.
        @test sort!([n for (n, _, a, r) in SAO_CURATED for (fn, w) in (("take_$n", a), ("make_$n", r)) if w === :trap]) ==
              ["c_ld"]
    end

    @testset "abi_fuzz records and functions" begin
        curated = Set(n for (n, _, _, _) in SAO_CURATED)
        checked = 0
        for c in SAO_CASES
            sao_subject(c.fn) in curated && continue
            if sao_agree(c.jlcs, c.llvm)
                @test true
            else
                @test (c.fn, c.pos, string(c.jlcs), c.family) == (c.fn, c.pos, string(c.llvm), nothing)
            end
            checked += 1
        end
        # A floor, not a count (the harness recounts): the generator still
        # makes a corpus worth the name.
        @test checked > 500
    end
end

end # requires
