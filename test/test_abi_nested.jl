# =============================================================================
# Nested-struct-member ABI trace test (library-free)
#
# Traces the C generator through compile → DWARF → wrap → live by-value
# crossings for structs with struct-typed members. Guards the SysV register
# classification contract: a 16-byte all-float struct travels in XMM registers;
# representing it as NTuple{16,UInt8} silently reads/writes integer registers.
# The probe runs in a subprocess so an ABI break can never take down the
# test session (see also: packed structs are expected to refuse loudly).
# =============================================================================

using Test

const ABI_TEST_DIR = joinpath(@__DIR__, "abi_nested_test")

@testset "Nested-struct ABI resolution" begin
    toml_path = joinpath(ABI_TEST_DIR, "replibuild.toml")

    # Regenerate the config each run (absolute paths, machine-local)
    write(toml_path, """
    [project]
    name = "abi_nested"
    root = "$(ABI_TEST_DIR)"

    [compile]
    flags = ["-O2", "-fPIC"]
    source_files = ["$(joinpath(ABI_TEST_DIR, "src", "nested.c"))"]
    include_dirs = ["$(joinpath(ABI_TEST_DIR, "include"))"]

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
    RepliBuild.wrap(toml_path)

    wrapper = joinpath(ABI_TEST_DIR, "julia", "AbiNested.jl")
    @test isfile(wrapper)

    # Subprocess-isolated probe: parse PROBE lines
    probe = joinpath(ABI_TEST_DIR, "probe_abi_nested.jl")
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $probe $wrapper`);
                        stdout=out, stderr=out))
    output = String(take!(out))
    println(output)

    @test proc.exitcode == 0
    @test occursin("PROBE_DONE", output)           # probe ran to completion (no abort)
    for probe_name in ["fields_resolved", "xform_return", "xform_byvalue_arg",
                       "mass_roundtrip", "disc_roundtrip", "poly_memory_class",
                       "nestint_roundtrip", "packed_byvalue_guard",
                       "float_param_loosening", "with_helper"]
        @test occursin(Regex("PROBE $(probe_name): PASS"), output)
    end
end
