# =============================================================================
# Convenience-overload ownership guard (library-free)
#
# The wrapper generators must NOT emit struct-by-value convenience overloads
# for Ptr{Struct} params: f(x::MyStruct) passed Ref(local copy) to the ccall,
# which is undefined behavior for any C function that frees, mutates-and-
# retains, or stores the pointer (crash-proven: cJSON_Delete(::cJSON) → glibc
# double-free abort). Ownership isn't recoverable from DWARF, so no name
# heuristic can gate the overload class safely — it is gone entirely; the base
# wrapper's ::Any params accept Ref(x)/pointers instead.
# Pinned survivors: the Vector{T} convenience path for input arrays, with
# Cstring returns aligned to the base wrapper's String policy.
# The live probe runs in a subprocess so a regression (double-free abort)
# can never take down the test session.
# =============================================================================

using Test
using RepliBuild

const CONV_TEST_DIR = joinpath(@__DIR__, "convenience_overload_test")

@testset "Convenience overloads: no struct-by-value footgun" begin
    toml_path = joinpath(CONV_TEST_DIR, "replibuild.toml")

    # Regenerate the config each run (absolute paths, machine-local)
    write(toml_path, """
    [project]
    name = "gripkit"
    root = "$(CONV_TEST_DIR)"

    [compile]
    flags = ["-O2", "-fPIC"]
    source_files = ["$(joinpath(CONV_TEST_DIR, "src", "grip.c"))"]
    include_dirs = ["$(joinpath(CONV_TEST_DIR, "include"))"]

    [link]
    # LTO off: trace the pure Tier-3 ccall path (Hub production configuration)
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
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)

    # Static shape: the by-value emission class is gone, the array class stays
    txt = read(wrapper, String)
    @test !occursin("accepts structs directly", txt)   # the footgun's marker comment
    @test occursin("accepts arrays directly", txt)     # Vector path survives

    # Subprocess-isolated live probe: parse PROBE lines
    probe = joinpath(CONV_TEST_DIR, "probe_convenience.jl")
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $probe $wrapper`);
                        stdout=out, stderr=out))
    output = String(take!(out))
    println(output)

    @test proc.exitcode == 0
    @test occursin("PROBE_DONE", output)               # probe ran to completion (no abort)
    for probe_name in ["no_byvalue_overload", "byvalue_call_refused",
                       "pointer_lifecycle", "vector_convenience",
                       "cstring_policy_aligned"]
        @test occursin(Regex("PROBE $(probe_name): PASS"), output)
    end
end
