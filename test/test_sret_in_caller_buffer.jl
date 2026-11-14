# =============================================================================
# A pass-by-reference return is built in the caller's buffer, live (2026-11-13)
#
# A record that is non-trivial for the purposes of calls (DW_CC_pass_by_reference)
# comes back through a hidden pointer, and C++ constructs it there. The Tier 2
# thunk used to pass its own alloca as that pointer and return the bytes, which
# MLIR's C-interface wrapper copied into the caller's buffer: a bytewise move,
# which C++ forbids for such an object. It held only for records with no pointer
# into themselves; libstdc++'s short std::string came back pointing at the
# thunk's dead frame. Now the thunk forwards the caller's buffer from argument
# slot 0 (`FunctionGen.returns_in_caller_buffer`, `thunk_symbol`).
#
# sret_inplace_test's SelfRef stores its own address; returned by value, that
# address must be the buffer the caller passed. Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const SRET_DIR = joinpath(@__DIR__, "sret_inplace_test")

if requires("A pass-by-reference return is built in the caller's buffer", :libJLCS, :clangxx)
@testset "A pass-by-reference return is built in the caller's buffer" begin
    toml_path = joinpath(SRET_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "sret_inplace"
    root = "$(escape_string(SRET_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(SRET_DIR, "src", "sret.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(SRET_DIR, "include")))"]

    [link]
    enable_lto = false
    optimization_level = "0"

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
    RepliBuild.build(toml_path)
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)
    md = RepliBuild._read_json(joinpath(dirname(wrapper), "compilation_metadata.json"))
    @test get(get(md["record_abi"], "SelfRef", Dict()), "pass", nothing) == "reference"

    r = run_isolated(`$(joinpath(SRET_DIR, "probe_sret.jl")) $wrapper`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    names = ["built_in_the_callers_buffer", "stl_string_built_in_the_callers_buffer",
             "wrapper_call_site", "stale_call_says_rewrap",
             "trivial_record_unchanged"]
    for n in names
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test count(n -> haskey(v, n), names) == length(names)
end
end # requires
