# =============================================================================
# A 16-byte SIMD vector in a union, by value, live (2026-11-16)
#
# Jolt's Vec3, Vec4, Quat and UVec4 are `union { __m128 mValue; float
# mF32[4]; }` (or the integer twin), passed and returned by value everywhere.
# The SysV classifier refused any record holding a 16-byte vector as "one SSE
# register no call statement states", and about 1,000 Jolt functions were ABI
# traps. One register is the answer for a vector ALONE. In a union the
# eightbyte merge decides: against the lanes' own floats it is (SSE, SSE),
# two registers ({ double, double } in clang's IR); against integers it is
# (INTEGER, INTEGER). Both are classes the thunks already state.
#
# The classification is diffed against LLVM's ABI library in
# test_sysv_abi_oracle.jl. This builds test/simd_union_test with the system
# compiler, wraps it, and passes the records through the thunks from a child,
# checking every lane. x86-64 SysV only: Win64 passes these by reference and
# AAPCS64 refuses a record holding a vector. Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const SIMD_UNION_DIR = joinpath(@__DIR__, "simd_union_test")

if RepliBuild.host_record_abi() !== :sysv
    @testset "A 16-byte SIMD vector in a union, live" begin
        @info "x86-64 SysV only; this host is $(RepliBuild.host_record_abi())"
        @test_skip false
    end
elseif requires("a 16-byte SIMD vector in a union, live", :libJLCS, :clangxx)
@testset "A 16-byte SIMD vector in a union, live" begin
    toml_path = joinpath(SIMD_UNION_DIR, "replibuild.toml")
    src = joinpath(SIMD_UNION_DIR, "simd_union.cpp")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "simdunion"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(SIMD_UNION_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-O2", "-fstandalone-debug"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(SIMD_UNION_DIR))"]

    [link]
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(SIMD_UNION_DIR, "julia", "Simdunion.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)
    md = RepliBuild._read_json(joinpath(SIMD_UNION_DIR, "julia", "compilation_metadata.json"))

    @testset "the classes, from this build's metadata" begin
        FG = RepliBuild.JLCSIRGenerator.FunctionGen
        sd = md["struct_definitions"]
        cls(name) = (v = FG.sysv_record_abi(name, sd); (v.kind, v.classes))
        @test cls("Vec4f") == (:regs, [:sse, :sse])          # the float union: two SSE registers
        @test cls("Vec4u") == (:regs, [:int, :int])          # the integer union: two general registers
        @test cls("Quatf") == (:regs, [:sse, :sse])          # the union one record down
        @test cls("Lone")[1] === :unstated                   # a vector alone: one SSE register
    end

    @testset "only the lone vector is a trap" begin
        traps = [m.captures[1] for m in eachmatch(r"ABI Safety Trap: cannot call '(\w+)'", code)]
        @test sort(unique(traps)) == ["lone_x"]
        for f in ("v_add", "v_lane", "v_mixed", "u_or", "u_lane", "q_conj", "q_w", "Body_getPos", "Body_setPos", "Body_scaled")
            @test (f, occursin(Regex("\\nfunction $f\\("), code)) == (f, true)
        end
    end

    @testset "every lane, through the thunks, from a child" begin
        r = run_isolated(`$(joinpath(SIMD_UNION_DIR, "probe_simd_union.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["float_union_by_value_both_ways", "float_union_among_other_arguments", "integer_union_by_value",
                 "nested_record_by_value", "member_functions", "lone_vector_still_refused"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
