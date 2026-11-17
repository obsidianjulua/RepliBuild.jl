# =============================================================================
# Enumerators above typemax(Int64), live (2026-11-16)
#
# Jolt's `LargeIslandSplitter::EIterationStatus : uint64` has the enumerator
# 0xffff000000000000. The bitflag test converted every value to Int64, threw
# on it, and that one enum ended the whole wrap of the library. The unit
# checks of `_is_bitflag_enum` are in test_c_generator_policies.jl; this
# builds test/wide_enum_test, wraps it, and passes the values through the
# library from a child. Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const WIDE_ENUM_DIR = joinpath(@__DIR__, "wide_enum_test")

if requires("enumerators above typemax(Int64), live", :libJLCS, :clangxx)
@testset "Enumerators above typemax(Int64), live" begin
    toml_path = joinpath(WIDE_ENUM_DIR, "replibuild.toml")
    src = joinpath(WIDE_ENUM_DIR, "src", "wide_enum.cpp")
    inc = joinpath(WIDE_ENUM_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "wideenum"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(WIDE_ENUM_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-O2", "-fstandalone-debug"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(inc))"]

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
    # The wrap is the step that threw.
    wrapped = try
        RepliBuild.wrap(toml_path); true
    catch e
        @info "wrap threw" exception = e
        false
    end
    @test wrapped
    wrapper = joinpath(WIDE_ENUM_DIR, "julia", "Wideenum.jl")
    @test isfile(wrapper)
    code = isfile(wrapper) ? read(wrapper, String) : ""

    @testset "the enums as emitted" begin
        # Masks beside shifts: an open enum over its 64-bit underlying type.
        @test occursin("primitive type Status <: RepliBuild.OpenEnums.OpenEnum{UInt64} 64 end", code)
        @test occursin("const StatusIterationMask = Status(18446462598732840960)", code)
        # A flag set whose top bit is 63, and an ordinary one: integer aliases.
        @test occursin("\nconst Wide = UInt64\n", code)
        @test occursin("const WideTop = Wide(9223372036854775808)", code)
        @test occursin("\nconst Small = Cuint\n", code)
    end

    @testset "the values, from a child" begin
        r = run_isolated(`$(joinpath(WIDE_ENUM_DIR, "probe_wide_enum.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["value_above_int64_round_trips", "flag_set_through_bit_63", "small_flag_set_unchanged"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
