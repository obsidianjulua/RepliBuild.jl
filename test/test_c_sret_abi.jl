# =============================================================================
# Tier 3 by-value record returns, live (C1)
#
# Compiles c_sret_test, wraps it as C and calls every function in a child
# process, checking each argument member the callee read and each member of the
# record it returned. C1 is the first two: abi_fuzz seed 36's `f4` and seed 3's
# `f2` return `{ char m1; short m2[3]; }`, an 8-byte register-class record the
# generator used to return through a hidden pointer the callee never reads.
#
# The probe runs this host's convention only. The second testset checks the
# classifier against clang's own lowering of the same file for BOTH targets,
# using this build's DWARF metadata: clang marks a hidden-pointer return
# `sret`, and `_c_return_in_memory` must agree for SysV and for Win64. That is a
# specification check; running the SysV side needs a Linux host.
#
# Needs the C bucket (a build) and, for the second testset, clang on PATH.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const C_SRET_DIR = joinpath(@__DIR__, "c_sret_test")

# Every function in c_sret.h, and the ones that return a record by value.
const C_SRET_FUNCTIONS = ["f4", "f2", "make_ic12", "ic12_sum", "make_p4", "make_p8",
                          "take_p8", "make_big24", "make_is20", "make_pk17", "make_out6", "make_bf12"]
const C_SRET_RECORD_RETURNS = ["f4", "f2", "make_ic12", "make_p4", "make_p8",
                               "make_big24", "make_is20", "make_pk17", "make_out6", "make_bf12"]

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
    end # requires
end
