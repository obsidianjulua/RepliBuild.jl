# =============================================================================
# AAPCS64 call shapes against clang's lowering for aarch64-linux-gnu
#
# FunctionGen classifies a record passed or returned by value from its DWARF
# facts (`aapcs64_record_abi`, src/IRGen/ir_gen/AAPCS64RecordABI.jl) and
# states the verdict on the call (`aapcs64_args`, `aapcs64_ret`); buildCallShape
# (src/mlir/impl/JLCSPasses.cpp) decides the call from it under
# AbiTarget::AAPCS64: a pointer to a copy, an sret slot, or one block of
# registers (`i64`, `[2 x i64]`, `i128`, `iN`, `[N x fT]`). This test asks the
# dialect and clang about the same records and lists every disagreement.
#
#   corpus  one C unit of hand-written records (AAS_RECORDS), each with
#           `void take_R(R)` and `R make_R(void)`, plus functions mixing them.
#           Built once as C on THIS host; the DWARF facts come back in
#           compilation_metadata.json. The records' layouts are the same on
#           x86-64 and AArch64 (both LP64, `long double` 16 bytes).
#   JLCS    the module `generate_jlcs_ir` writes for that metadata with
#           `abi = :aapcs64`, every jlcs.ffe_call shaped by the real
#           buildCallShape under the AArch64 target
#           (`jlcs_probe_aapcs64_call_shapes`), on any host.
#   clang   `clang --target=aarch64-linux-gnu -S -emit-llvm` of the same unit:
#           the coerced parameter and return types of every `define`.
#
# This is a SPECIFICATION test (like test_win64_abi.jl), not a behavioural one:
# nothing here runs on an AArch64 host. It catches an encoded rule that
# disagrees with clang, and an alignment the two sides read differently. The
# live run is the arm64 job of .github/workflows/tier2-linux.yml.
#
# Alignment is compared on the register side only (an `i128` pair against
# `[2 x i64]`): a copy or sret slot the dialect aligns to at least 8 where
# clang stamps the record's natural alignment, and no AArch64 load checks it.
#
# Records the generator refuses by design (AAS_REFUSED: a SIMD vector member,
# which AAPCS64 may pass as a homogeneous vector aggregate) must be exactly the
# ones with no thunk, and clang must indeed give them a vector register.
#
# Needs libJLCS and a clang that can target aarch64-linux-gnu (any clang with
# the AArch64 target registered: IR generation needs no sysroot). Skips
# elsewhere.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const AAS_FG = RepliBuild.JLCSIRGenerator.FunctionGen
const AAS_MN = RepliBuild.MLIRNative
const AAS_TRIPLE = "aarch64-linux-gnu"
const AAS_CLANG = Sys.which("clang")

# A clang without the AArch64 target cannot act as the specification.
function aas_can_target()
    AAS_CLANG === nothing && return false
    try
        mktempdir() do d
            p = joinpath(d, "probe.c")
            write(p, "int f(void){return 0;}\n")
            # A real output file, not /dev/null (see test_win64_abi.jl).
            success(`$AAS_CLANG --target=$AAS_TRIPLE -S -emit-llvm -o $(joinpath(d, "probe.ll")) $p`)
        end
    catch
        false
    end
end

# ── The corpus ───────────────────────────────────────────────────────────────
# (name, C definition). One per rule of AAPCS64RecordABI.jl, and the shapes
# clang 23 was measured on when the rules were written.
const AAS_RECORDS = [
    # homogeneous floating-point aggregates, at any size
    ("HFA2",   "typedef struct { float a, b; } HFA2;"),
    ("HFA4",   "typedef struct { float a; float b[3]; } HFA4;"),
    ("HFAD4",  "typedef struct { double a, b, c, d; } HFAD4;"),
    ("D3",     "typedef struct { double a, b, c; } D3;"),
    ("D1",     "typedef struct { double d; } D1;"),
    ("F1",     "typedef struct { float f; } F1;"),
    ("NHFA",   "typedef struct { struct { float a, b; } in; float c; } NHFA;"),
    ("AIN",    "typedef struct { F1 in[2]; } AIN;"),
    ("CF",     "typedef struct { _Complex float z; } CF;"),
    ("CD",     "typedef struct { _Complex double z; } CD;"),
    ("UFF",    "typedef union { float f; float g; } UFF;"),
    ("HALF3",  "typedef struct { _Float16 h[3]; } HALF3;"),
    ("LD",     "typedef struct { long double x; } LD;"),
    ("LD2",    "typedef struct { long double a, b; } LD2;"),
    ("CLD",    "typedef struct { _Complex long double z; } CLD;"),   # std::complex<long double>
    # what breaks an HFA
    ("HFA5",   "typedef struct { float a, b, c, d, e; } HFA5;"),
    ("HFAPAD", "typedef struct { float a; float b; char pad; } HFAPAD;"),
    ("MIX",    "typedef struct { double a; float b; } MIX;"),
    ("FD",     "typedef struct { float a; double b; } FD;"),
    ("FI",     "typedef struct { float a; int b; } FI;"),
    ("BFF",    "typedef struct { int a : 3; float f; } BFF;"),
    ("UFI",    "typedef union { float f; int i; } UFI;"),
    # general registers by size; field alignment unread
    ("S8",     "typedef struct { char a; int b; } S8;"),
    ("P5",     "typedef struct __attribute__((packed)) { char a; int b; } P5;"),
    ("PCD",    "typedef struct __attribute__((packed)) { char c; double d; } PCD;"),
    ("S12",    "typedef struct { int a, b, c; } S12;"),
    ("S16",    "typedef struct { long long a, b; } S16;"),
    ("S9",     "typedef struct { long long a; char b; } S9;"),
    ("C16",    "typedef struct { char a[16]; } C16;"),
    ("SF",     "typedef struct { short s; float f; } SF;"),
    ("NEST8",  "typedef struct { struct { long long x; } in; } NEST8;"),
    # the even register pair, and the record's own `aligned` read by a return only
    ("MA16",   "typedef struct { _Alignas(16) long long a; } MA16;"),
    ("A16",    "typedef struct __attribute__((aligned(16))) { long long a; } A16;"),
    ("NA16",   "typedef struct { A16 a; } NA16;"),
    ("NMA16",  "typedef struct { MA16 m; } NMA16;"),
    ("I128",   "typedef struct { __int128 x; } I128;"),
    # over 16 bytes: a pointer to a copy, an sret slot
    ("S17",    "typedef struct { char a[17]; } S17;"),
    ("CA16",   "typedef struct { char c; A16 a; } CA16;"),
    ("CI128",  "typedef struct { char c; __int128 x; } CI128;"),
    ("BIG16",  "typedef struct __attribute__((aligned(16))) { long long a, b, c; } BIG16;"),
    ("FAL16",  "typedef struct { float a; _Alignas(16) float b; } FAL16;"),
    # nothing
    ("E0",     "typedef struct { } E0;"),
    # refused: a SIMD vector member
    ("V4",     "typedef struct { float v __attribute__((vector_size(16))); } V4;"),
    ("V2x2",   "typedef struct { float v __attribute__((vector_size(8))); float w __attribute__((vector_size(8))); } V2x2;"),
]
const AAS_REFUSED = Set(["V4", "V2x2"])

# Functions mixing records with scalars and each other, in argument order.
const AAS_MIXED = [
    "double mix_a(S12 a, HFA2 b, S17 c, int d) { (void)a; (void)b; (void)c; return d; }",
    "double mix_b(A16 a, MA16 b, double x) { (void)a; (void)b; return x; }",
    "double mix_c(HFAD4 a, E0 e, LD b, char k) { (void)a; (void)e; (void)b; return k; }",
    "double mix_d(P5 a, UFI b, CI128 c, I128 d, NHFA e) { (void)a; (void)b; (void)c; (void)d; (void)e; return 1.0; }",
]

function aas_corpus()
    io = IOBuffer()
    # No system header: the same text is compiled for aarch64-linux-gnu, and a
    # multiarch glibc (Debian, Ubuntu) keeps its target-specific headers under
    # a directory a bare `--target` cannot find (`bits/libc-header-start.h`).
    # The builtin needs none.
    println(io, "/* AAPCS64 specification corpus, generated by test/test_aapcs64_abi_spec.jl */\n")
    for (_, def) in AAS_RECORDS
        println(io, def)
    end
    println(io)
    for (name, _) in AAS_RECORDS
        println(io, "void take_$(name)($(name) x) { (void)x; }")
        println(io, "$(name) make_$(name)(void) { $(name) r; __builtin_memset(&r, 0, sizeof r); return r; }")
    end
    println(io)
    for f in AAS_MIXED
        println(io, f)
    end
    return String(take!(io))
end

"Build the corpus as C in `dir`; the metadata it wrote."
function aas_build(dir::String, src::String)
    mkpath(joinpath(dir, "src"))
    write(joinpath(dir, "src", "corpus.c"), src)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "aapcs64_spec_corpus"
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

# ── Verdicts ─────────────────────────────────────────────────────────────────
# Both answers reduced to where a record goes:
#   (:ignore,)              passed nowhere / returned as void
#   (:indirect,)            a pointer to a caller copy
#   (:sret,)                a hidden return pointer
#   (:i128,)                an even general register pair
#   (:int, bits)            one general register of that width (i64 for an
#                           argument, iN for a return of N bits)
#   (:gpr, 2)               [2 x i64]
#   (:hfa, base, n)         [n x base]
#   (:hva, n)               [n x <k x T>] — what clang gives a vector record
#   (:scalar,)              not a record
#   (:odd, text)            neither side's vocabulary

const AAS_FP = Dict("half" => :f16, "float" => :f32, "double" => :f64, "fp128" => :f128,
                    "f16" => :f16, "f32" => :f32, "f64" => :f64, "f128" => :f128)

function aas_block(t::AbstractString)
    t = strip(t)
    t == "i128" && return (:i128,)
    m = match(r"^i(\d+)$", t)
    m !== nothing && return (:int, parse(Int, m.captures[1]))
    m = match(r"^(?:!llvm\.array<|\[)(\d+) x (\w+)(?:>|\])$", t)
    if m !== nothing
        n, el = parse(Int, m.captures[1]), m.captures[2]
        el == "i64" && return (:gpr, n)
        haskey(AAS_FP, el) && return (:hfa, AAS_FP[el], n)
    end
    m = match(r"^(?:!llvm\.array<|\[)(\d+) x (?:vector<|<)", t)
    m !== nothing && return (:hva, parse(Int, m.captures[1]))
    return (:odd, String(t))
end

# A probe field (`regs:<t>`, `indirect:<a>`, `sret:<a>`, `ignore`, `scalar:<t>`, `void`).
function aas_jlcs_verdict(field::AbstractString)
    field == "ignore" && return (:ignore,)
    field == "void" && return (:ignore,)
    startswith(field, "indirect:") && return (:indirect,)
    startswith(field, "sret:") && return (:sret,)
    startswith(field, "scalar:") && return (:scalar,)
    startswith(field, "regs:") && return aas_block(field[6:end])
    return (:odd, String(field))
end

# Split `s` at top-level commas (outside (), [], {}, <>).
function aas_split(s::AbstractString)
    out = String[]
    depth = 0
    start = 1
    for (i, c) in enumerate(s)
        if c in ('(', '[', '{', '<')
            depth += 1
        elseif c in (')', ']', '}', '>')
            depth -= 1
        elseif c == ',' && depth == 0
            push!(out, strip(s[start:i-1]))
            start = i + 1
        end
    end
    tail = strip(s[start:end])
    isempty(tail) || push!(out, tail)
    return out
end

# Everything in a parameter or return spelling that is not its type: the
# attributes clang stamps and the parameter's `%name`. Types start with `[`,
# `{`, `<`, `%struct.`/`%union.` or a keyword, never with a bare `%N`.
const AAS_IR_WORD_ATTR = r"\b(?:noundef|dso_local|zeroext|signext|nonnull|noalias|writable|dead_on_return|dead_on_unwind)\b"

function aas_clean(t::AbstractString)
    t = replace(strip(t), r"\s+%(?!struct\.|union\.)[\w.]+$" => "")   # the parameter's own name, last
    t = replace(t, r"alignstack\(\d+\)" => "", r"\balign \d+" => "", AAS_IR_WORD_ATTR => "")
    return String(strip(replace(t, r"\s+" => " ")))
end

# One clang parameter, as a verdict.
function aas_clang_param(p::AbstractString)
    occursin("sret(", p) && return (:sret,)
    t = aas_clean(p)
    startswith(t, "ptr") && return (:indirect,)
    t in ("double", "float", "half", "fp128", "i1", "i8", "i16", "i32") && return (:scalar,)
    return aas_block(t)
end

# The leaves of an LLVM type, through the module's type table (`%struct.X = type { … }`).
function aas_leaves!(out::Vector{String}, t::AbstractString, types::Dict{String,String}, depth::Int = 0)
    depth > 16 && return out
    t = strip(t)
    if startswith(t, "%")
        return aas_leaves!(out, types[t], types, depth + 1)
    elseif startswith(t, "{") || startswith(t, "<{")
        body = startswith(t, "<{") ? t[3:end-2] : t[2:end-1]
        for e in aas_split(body)
            aas_leaves!(out, e, types, depth + 1)
        end
        return out
    end
    m = match(r"^\[(\d+) x (.*)\]$", t)
    if m !== nothing
        for _ in 1:parse(Int, m.captures[1])
            aas_leaves!(out, m.captures[2], types, depth + 1)
        end
        return out
    end
    push!(out, t)
    return out
end

# A clang return type, as a verdict. A struct returned directly is an HFA. A
# `void` is the sret form when a parameter carries the slot, else nothing
# (every `take_` is void, and so is a `make_` of an ignored record).
function aas_clang_ret(t::AbstractString, params::Vector{String}, types::Dict{String,String})
    t = aas_clean(t)
    if t == "void"
        any(p -> occursin("sret(", p), params) && return (:sret,)
        return (:ignore,)
    end
    if startswith(t, "%")
        leaves = aas_leaves!(String[], t, types)
        if !isempty(leaves) && allequal(leaves) && haskey(AAS_FP, leaves[1]) && length(leaves) <= 4
            return (:hfa, AAS_FP[leaves[1]], length(leaves))
        end
        startswith(leaves[1], "<") && return (:hva, length(leaves))
        return (:odd, "direct $t = $(types[t])")
    end
    t in ("double", "float") && return (:scalar,)
    return aas_block(t)
end

"""
    aas_clang(src) -> Dict(name => (ret, args))

clang's lowering of every function in `src` for aarch64-linux-gnu, as verdicts.
`ret` is the function's return verdict; `args` has one verdict per record
parameter clang kept (an ignored empty record has none) and `(:scalar,)` for a
scalar.
"""
function aas_clang(src::String)
    ir = mktempdir() do d
        cfile = joinpath(d, "corpus.c")
        write(cfile, src)
        read(`$AAS_CLANG --target=$AAS_TRIPLE -O0 -S -emit-llvm -o - $cfile`, String)
    end
    types = Dict{String,String}()
    for m in eachmatch(r"(?m)^(%[\w.]+) = type (.*)$", ir)
        types[m.captures[1]] = m.captures[2]
    end
    out = Dict{String,Tuple{Any,Vector{Any}}}()
    for m in eachmatch(r"(?m)^define (.*?) @(\w+)\((.*)\)\s*(?:#\d+|\{)", ir)
        rettext, name, paramtext = m.captures
        params = aas_split(paramtext)
        ret = aas_clang_ret(rettext, params, types)
        args = Any[aas_clang_param(p) for p in params if !occursin("sret(", p)]
        out[name] = (ret, args)
    end
    return out
end

"""
    aas_jlcs(meta) -> (shapes, traps)

`shapes`: callee => (ret verdict, arg verdicts) from the dialect's AAPCS64 probe
over the module the generator writes for `meta` with `abi = :aapcs64`;
`traps`: callee => why the generator gave it no thunk.
"""
function aas_jlcs(meta)
    vt = RepliBuild.DWARFParser.VtableInfo(Dict{String,RepliBuild.DWARFParser.ClassInfo}(),
                                           Dict{String,UInt64}(), Dict{String,UInt64}())
    ir = RepliBuild.JLCSIRGenerator.generate_jlcs_ir(vt, meta; abi = :aapcs64)
    traps = Dict(String(m.captures[1]) => String(m.captures[2])
                 for m in eachmatch(r"(?m)^// no thunk for (\S+): (.*)$", ir))
    @test occursin("aapcs64_", ir)
    @test !occursin("sysv_args", ir)
    ctx = AAS_MN.create_context()
    try
        mod = AAS_MN.parse_module(ctx, ir)
        text = unsafe_string(ccall((:jlcs_probe_aapcs64_call_shapes, AAS_MN.libJLCS), Cstring,
                                   (AAS_MN.MlirModule,), mod))
        shapes = Dict{String,Tuple{Any,Vector{Any}}}()
        for line in split(text, '\n'; keepempty = false)
            parts = String.(split(line, '\t'))
            length(parts) >= 2 || continue
            ret = aas_jlcs_verdict(parts[2])
            # An ignored empty record is an operand the call drops; clang has no
            # parameter for it, so neither list keeps it.
            args = Any[aas_jlcs_verdict(f) for f in parts[3:end] if f != "ignore"]
            shapes[parts[1]] = (ret, args)
        end
        return shapes, traps
    finally
        AAS_MN.destroy_context(ctx)
    end
end

# Scalars drop out of both lists; what is left is every record crossing in order.
aas_records_only(args) = Any[a for a in args if a != (:scalar,)]

# ── The test ─────────────────────────────────────────────────────────────────

if requires("AAPCS64 call shapes vs clang's aarch64 lowering", :libJLCS, :clang)
    if !aas_can_target()
        @testset "AAPCS64 call shapes vs clang (skipped: clang cannot target $AAS_TRIPLE)" begin
            @test_skip false
        end
    else
        @testset "AAPCS64 call shapes vs clang's aarch64 lowering" begin
            src = aas_corpus()
            clang = aas_clang(src)
            @test length(clang) == 2 * length(AAS_RECORDS) + length(AAS_MIXED)

            dir = mktempdir()
            meta = aas_build(dir, src)
            shapes, traps = aas_jlcs(meta)

            # Every record the generator refused is one it refuses by design,
            # and every one it refuses by design is refused: nothing else is
            # missing a thunk.
            refused_fns = Set(vcat(["take_$r" for r in AAS_REFUSED], ["make_$r" for r in AAS_REFUSED]))
            @test Set(keys(traps)) == refused_fns
            for f in sort!(collect(refused_fns))
                @test (f, occursin("SIMD vector", get(traps, f, ""))) == (f, true)
                @test (f, haskey(shapes, f)) == (f, false)
                # …and clang does put such a record in vector registers.
                cr, ca = clang[f]
                vec = startswith(f, "take_") ? ca : [cr]
                @test (f, any(v -> v[1] === :hva, vec)) == (f, true)
            end

            disagreements = String[]
            checked = 0
            for name in sort!(collect(keys(clang)))
                name in refused_fns && continue
                @test haskey(shapes, name)
                haskey(shapes, name) || continue
                (jr, ja), (cr, ca) = shapes[name], clang[name]
                jrec, crec = aas_records_only(ja), aas_records_only(ca)
                if jr != cr
                    push!(disagreements, "$name return: JLCS $(jr), clang $(cr)")
                end
                if jrec != crec
                    push!(disagreements, "$name arguments: JLCS $(jrec), clang $(crec)")
                end
                # Scalars sit at the same positions too (an ignored record
                # shifts nothing on either side).
                jr_pos = [i for (i, a) in enumerate(ja) if a == (:scalar,)]
                cr_pos = [i for (i, a) in enumerate(ca) if a == (:scalar,)]
                jr_pos == cr_pos || push!(disagreements, "$name scalar positions: JLCS $(jr_pos), clang $(cr_pos)")
                checked += 1
            end
            @test checked == length(clang) - length(refused_fns)
            @test checked >= 2 * (length(AAS_RECORDS) - length(AAS_REFUSED))
            isempty(disagreements) || @error "AAPCS64 call shapes disagree with clang" disagreements
            @test isempty(disagreements)

            # The spot checks the rules were written from, so a probe that
            # went silent (every function `(:odd, …)` on both sides) cannot
            # pass as agreement.
            @test shapes["make_P5"][1] == (:int, 40)
            @test shapes["take_A16"][2] == [(:gpr, 2)]
            @test shapes["make_A16"][1] == (:i128,)
            @test shapes["take_MA16"][2] == [(:i128,)]
            @test shapes["take_HFAD4"][2] == [(:hfa, :f64, 4)]
            @test shapes["make_LD"][1] == (:hfa, :f128, 1)
            @test shapes["take_S17"][2] == [(:indirect,)]
            @test shapes["make_S17"][1] == (:sret,)
            @test shapes["take_E0"][2] == []
            @test shapes["mix_a"][2] == [(:gpr, 2), (:hfa, :f32, 2), (:indirect,), (:scalar,)]
        end
    end
end # requires
