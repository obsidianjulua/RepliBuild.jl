# =============================================================================
# Tier 3 by-value record returns, live (C1)
#
# Compiles c_sret_test, wraps it as C and calls every function in a child
# process, checking each argument member the callee read and each member of the
# record it returned. C1 is the first two: abi_fuzz seed 36's `f4` and seed 3's
# `f2` return `{ char m1; short m2[3]; }`, an 8-byte register-class record the
# generator used to return through a hidden pointer the callee never reads.
#
# The `make_*` records after BF12 are more aligned than the Julia type standing
# in for them, and each maker stores its return with movaps: the wrapper must
# hand the callee a buffer at the record's alignment, which Julia's own lowering
# of the stand-in does not.
#
# The `take_*` functions pass those records by value after a 24-byte argument
# that lives on the stack. C places a MEMORY-class record at its own alignment
# past it, and a byval at the stand-in's alignment (8) read `take_d4a`'s record
# one slot early (W2, 2026-10-24). The build records each definition's `byval`
# slot alignment off the linked IR (`byval_params`), and the wrapper passes a
# 16-aligned record inside `_RB_ByVal16`, refuses what it cannot place (A32,
# 32-aligned), and leaves a register-class record (V4A) to the plain ccall.
#
# The probe runs this host's convention only. The second testset checks the
# classifier against clang's own lowering of the same file for BOTH targets,
# using this build's DWARF metadata: clang marks a hidden-pointer return
# `sret`, and `_c_return_in_memory` must agree for SysV and for Win64. The third
# checks `_c_record_align` against the alignment clang gives each record's
# global. Both are specification checks; running the SysV side needs a Linux
# host.
#
# Needs the C bucket (a build) and, for the second testset, clang on PATH.
# =============================================================================

using Test
using RepliBuild
import Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const C_SRET_DIR = joinpath(@__DIR__, "c_sret_test")

# Every function in c_sret.h, and the ones that return a record by value.
const C_SRET_ALIGNED = ["make_ab16", "make_v4a", "make_d4a", "make_ma32", "make_a32",
                        "make_hvec", "make_hv4", "make_nesta"]
# The same records as arguments, after a 24-byte stack argument (W2).
const C_SRET_TAKES = ["take_ab16", "take_v4a", "take_d4a", "take_ma32", "take_a32",
                      "take_hvec", "take_hv4", "take_nesta"]
const C_SRET_FUNCTIONS = ["f4", "f2", "make_ic12", "ic12_sum", "make_p4", "make_p8",
                          "take_p8", "make_big24", "make_is20", "make_pk17", "make_out6", "make_bf12",
                          "make_tg16", "take_tg16", C_SRET_ALIGNED..., C_SRET_TAKES...]
const C_SRET_RECORD_RETURNS = ["f4", "f2", "make_ic12", "make_p4", "make_p8",
                               "make_big24", "make_is20", "make_pk17", "make_out6", "make_bf12", "make_tg16",
                               C_SRET_ALIGNED...]

# One function's body in a generated wrapper (its `function name(` up to `end`).
function c_sret_body(wsrc::AbstractString, name::AbstractString)
    i = findfirst("function $name(", wsrc)
    i === nothing && return ""
    r = SubString(wsrc, first(i))
    j = findfirst(r"\nend\b", r)
    return j === nothing ? String(r) : String(r[1:last(j)])
end

"""
    c_sret_aapcs64_shapes(wsrc)

What GeneratorC emits for this fixture under AAPCS64 (`abi = :aapcs64`), pinned
when the convention was built (2026-11-09): registers at 16 bytes or less and
HFAs at any size through Julia's own lowering, so every record return is a
plain `ccall` (no explicit sret, no aligned buffer: an HFA over 16 bytes, D4A,
comes back in registers, and no AArch64 load checks a buffer's alignment); a
pointer to a copy above 16 bytes with no carrier (`_RB_ByVal16` is never
emitted); the float blobs AB16 and BF12 called, since their facts (a bitfield
in each) rule an HFA out and they travel by size like their blobs, except
`take_ab16`, whose 16-aligned 16 bytes AAPCS64 puts in an even register pair
the blob cannot reach (`_c_register_pair_misplaced`); and every record holding
a SIMD vector refused. Asserted on the host's wrapper on an AArch64 host and on
a wrapper generated for AAPCS64 everywhere.
"""
function c_sret_aapcs64_shapes(wsrc::AbstractString)
    body(name) = c_sret_body(wsrc, name)
    plain = Dict("take_d4a" => "(Big24, D4A,), p, v)", "take_ma32" => "(Big24, MA32,), p, v)",
                 "take_nesta" => "(Big24, NestA,), p, v)", "take_a32" => "(Big24, A32,), p, v)",
                 "take_v4a" => "(Big24, V4A,), p, v)", "take_hv4" => "(Big24, HV4,), p, v)", "take_p8" => "(P8,), v)",
                 "take_tg16" => "(TG16,), v)")
    checked = 0
    for (f, sig) in sort!(collect(plain); by = first)
        @test (f, occursin("ccall((:$f, LIBRARY_PATH), ", body(f)) && occursin(sig, body(f))) == (f, true)
        checked += 1
    end
    for f in ("make_v4a", "make_d4a", "make_ma32", "make_a32", "make_hv4", "make_nesta", "make_ab16", "make_bf12",
              "make_tg16", "make_p4", "make_p8", "make_out6", "make_pk17", "make_big24", "make_is20", "make_ic12",
              "f2", "f4")
        b = body(f)
        @test (f, occursin("ccall((:$f, LIBRARY_PATH), ", b) && !occursin("ret_ptr", b) && !occursin("ret_buf", b)) ==
              (f, true)
        checked += 1
    end
    @test occursin("even-numbered register pair", body("take_ab16"))
    @test !occursin("AAPCS64 register window", wsrc)      # the facts settle every float blob here
    checked += 1
    for f in ("take_hvec", "make_hvec")
        @test (f, occursin("holds a SIMD vector", body(f))) == (f, true)
        checked += 1
    end
    @test checked == 29
    @test !occursin("_rb_byval16(", wsrc)
    @test !occursin("ABI Safety Trap", body("take_a32"))      # no slot to align: a plain call
 end

@testset "Tier 3 record returns, live (C1)" begin
    toml_path = joinpath(C_SRET_DIR, "replibuild.toml")
    src = joinpath(C_SRET_DIR, "src", "c_sret.c")
    inc = joinpath(C_SRET_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "c_sret"
    # escape_string on every interpolated path: TOML basic strings take escapes,
    # so a Windows `C:\\Users\\...` would parse as `\\U` and fail.
    root = "$(escape_string(C_SRET_DIR))"

    [compile]
    flags = ["-O2", "-fPIC"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(inc))"]

    [link]
    # LTO off: the pure Tier-3 ccall path (Hub production configuration)
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "c"

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true

    [cache]
    enabled = false
    """)

    RepliBuild.build(toml_path)
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(C_SRET_DIR, "julia", "CSret.jl")
    @test isfile(wrapper)

    @testset "every argument and return member, this host ($(RepliBuild.Wrapper._c_record_abi()))" begin
        r = run_isolated(`$(joinpath(C_SRET_DIR, "probe_c_sret.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        checked = 0
        for f in C_SRET_FUNCTIONS
            @test (f, get(v, f, "no PROBE line")) == (f, "PASS")
            checked += 1
        end
        @test checked == length(C_SRET_FUNCTIONS)
    end

    @testset "the wrapper's shape for the over-aligned arguments, this host" begin
        wsrc = read(wrapper, String)      # not `src`: that is the C file the clang testsets compile
        body(name) = let i = findfirst("function $name(", wsrc)
            i === nothing ? "" : (r = SubString(wsrc, first(i)); j = findfirst(r"\nend\b", r);
                                  j === nothing ? String(r) : String(r[1:last(j)]))
        end
        host_abi = RepliBuild.Wrapper._c_record_abi()
        if host_abi === :aapcs64
            c_sret_aapcs64_shapes(wsrc)
        elseif host_abi === :sysv
            checked = 0
            for f in ("take_d4a", "take_ma32", "take_hvec", "take_hv4", "take_nesta")
                @test (f, occursin("_rb_byval16(v, _RB_ByVal16{2})", body(f))) == (f, true)
                @test (f, occursin("(Big24, _RB_ByVal16{2},), p, _rb_byval16(v, _RB_ByVal16{2}))", body(f))) == (f, true)
                checked += 1
            end
            @test checked == 5
            @test occursin("ABI Safety Trap", body("take_a32"))
            @test occursin("32-byte-aligned stack slot", body("take_a32"))
            @test occursin("ABI Safety Trap", body("take_ab16"))          # SSE then INTEGER: the blob cannot claim it
            @test occursin("ccall((:take_v4a, LIBRARY_PATH), Cfloat, (Big24, V4A,), p, v)", body("take_v4a"))
            # INTEGER, INTEGER by the facts, whatever the float members: plain calls, as the blob travels.
            @test occursin("ccall((:make_bf12, LIBRARY_PATH), BF12, (Cfloat, Cint, Cint,)", body("make_bf12"))
            @test occursin("ccall((:make_tg16, LIBRARY_PATH), TG16, (Cint, Cdouble,)", body("make_tg16"))
            @test occursin("ccall((:take_tg16, LIBRARY_PATH), Cdouble, (TG16,), v)", body("take_tg16"))
            @test !occursin("ABI Safety Trap", body("make_bf12"))
            @test count("struct _RB_ByVal16{N}", wsrc) == 1
        else
            # Win64 passes each of these by reference to a copy aligned as the
            # record is, and Julia's copy is aligned as the stand-in: every
            # record aligned above 8 crosses inside the carrier (2026-11-01),
            # the 32-aligned A32 included, since 16 is all Julia can give.
            checked = 0
            for f in ("take_d4a", "take_ma32", "take_hvec", "take_hv4", "take_nesta", "take_a32")
                @test (f, occursin("_rb_byval16(v, _RB_ByVal16{2})", body(f))) == (f, true)
                @test (f, occursin("(Big24, _RB_ByVal16{2},), p, _rb_byval16(v, _RB_ByVal16{2}))", body(f))) == (f, true)
                checked += 1
            end
            @test checked == 6
            @test occursin("_rb_byval16(v, _RB_ByVal16{1})", body("take_v4a"))     # 16 bytes: by reference too
            @test count("struct _RB_ByVal16{N}", wsrc) == 1
            # Big24 is 8-aligned: Julia's own copy, as before.
            @test occursin("(Big24,", body("take_d4a")) && !occursin("_rb_byval16(p", body("take_d4a"))
        end
    end

    @testset "the wrapper's shape for AAPCS64, from this host's metadata" begin
        # The same metadata generated for AArch64: what an AAPCS64 host's
        # wrapper looks like, pinned on every host (`c_sret_aapcs64_shapes`).
        W = RepliBuild.Wrapper
        cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
        meta = RepliBuild._read_json(joinpath(C_SRET_DIR, "julia", "compilation_metadata.json"))
        libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
        code = W.generate_introspective_module_c(cfg, libref, meta, "CSretA64", W.create_type_registry(cfg), false;
                                                 abi = :aapcs64)
        c_sret_aapcs64_shapes(code)
    end

    if requires("C record return class vs clang", :clang)
        @testset "return class agrees with clang, SysV and Win64" begin
            W = RepliBuild.Wrapper
            meta = RepliBuild._read_json(joinpath(C_SRET_DIR, "julia", "compilation_metadata.json"))
            sd = meta["struct_definitions"]
            rets = Dict(String(f["name"]) => String(f["return_type"]["c_type"]) for f in meta["functions"])
            checked = 0
            for (triple, abi) in (("x86_64-unknown-linux-gnu", :sysv), ("x86_64-w64-windows-gnu", :win64))
                ll = joinpath(mktempdir(), "c_sret.ll")
                @test success(`$(Sys.which("clang")) --target=$triple -O1 -S -emit-llvm -I$inc -o $ll $src`)
                defs = Dict(String(m.captures[1]) => String(m.match)
                            for m in eachmatch(r"(?m)^define [^\n]*?@(\w+)\([^\n]*$", read(ll, String)))
                for f in C_SRET_RECORD_RETURNS
                    info = sd[rets[f]]
                    want = occursin(r"\bsret\b", get(defs, f, ""))
                    got = W._c_return_in_memory(W._parse_dwarf_size(info), W._c_has_unaligned_field(info, sd); abi)
                    @test (f, abi, got) == (f, abi, want)
                    checked += 1
                end
            end
            @test checked == 2 * length(C_SRET_RECORD_RETURNS)
        end

        @testset "record alignment agrees with clang, SysV and Win64" begin
            W = RepliBuild.Wrapper
            meta = RepliBuild._read_json(joinpath(C_SRET_DIR, "julia", "compilation_metadata.json"))
            sd = meta["struct_definitions"]
            checked = 0
            for triple in ("x86_64-unknown-linux-gnu", "x86_64-w64-windows-gnu")
                ll = joinpath(mktempdir(), "c_sret.ll")
                @test success(`$(Sys.which("clang")) --target=$triple -O1 -S -emit-llvm -I$inc -o $ll $src`)
                # `@g_v4a = dso_local global %struct.V4A zeroinitializer, align 16`
                globals = Dict(String(m.captures[1]) => parse(Int, m.captures[2])
                               for m in eachmatch(r"(?m)^@g_\w+ = [^\n]*?global %struct\.(\w+) [^\n]*, align (\d+)\r?$",
                                                  read(ll, String)))
                for (rec, want) in sort(collect(globals))
                    @test (rec, triple, W._c_record_align(sd[rec], sd)) == (rec, triple, want)
                    checked += 1
                end
            end
            @test checked == 2 * length(C_SRET_ALIGNED)
        end

        @testset "byval slot alignments: the build records what clang lowers" begin
            C = RepliBuild.Compiler
            meta = RepliBuild._read_json(joinpath(C_SRET_DIR, "julia", "compilation_metadata.json"))
            fns = Dict(String(f["name"]) => f for f in meta["functions"])
            # SysV: a MEMORY-class record is `byval` at its alignment; a
            # register-class one (V4A, AB16) and an sret return are not. Win64
            # passes records by reference and marks nothing `byval`.
            want = Dict("take_ab16" => [("Big24", 8)], "take_v4a" => [("Big24", 8)],
                        "take_d4a" => [("Big24", 8), ("D4A", 16)], "take_ma32" => [("Big24", 8), ("MA32", 16)],
                        "take_a32" => [("Big24", 8), ("A32", 32)], "take_hvec" => [("Big24", 8), ("HVec", 16)],
                        "take_hv4" => [("Big24", 8), ("HV4", 16)], "take_nesta" => [("Big24", 8), ("NestA", 16)],
                        "take_p8" => [("P8", 8)], "f2" => [], "make_d4a" => [])
            pairs_of(facts) = [(String(f["record"]), Int(f["align"])) for f in facts]
            checked = 0
            for (triple, abi) in (("x86_64-unknown-linux-gnu", :sysv), ("x86_64-w64-windows-gnu", :win64))
                ll = joinpath(mktempdir(), "c_sret.ll")
                @test success(`$(Sys.which("clang")) --target=$triple -O1 -S -emit-llvm -I$inc -o $ll $src`)
                defs = C._byval_definitions([ll])
                for (f, w) in sort(collect(want))
                    @test (f, abi, haskey(defs, f)) == (f, abi, true)
                    @test (f, abi, pairs_of(get(defs, f, []))) == (f, abi, abi === :sysv ? w : [])
                    checked += 1
                end
            end
            # This build's metadata, read off the IR the library was linked
            # from, says the same for this host.
            host = RepliBuild.Wrapper._c_record_abi()
            for (f, w) in sort(collect(want))
                @test (f, haskey(fns[f], "byval_params")) == (f, true)
                @test (f, pairs_of(get(fns[f], "byval_params", []))) == (f, host === :sysv ? w : [])
                checked += 1
            end
            @test checked == 3 * length(want)
        end
    end # requires
end
