# =============================================================================
# C++ by-value record routing, live
#
# Builds cpp_routing_test, a C++ library whose extern "C" functions pass and
# return records by value, wraps it, and checks two things: which tier each
# function went to, and that every one of them really works (a child process
# calls each and checks every member, in both directions).
#
#   * natural records whose Julia type is the C layout (CS3, Mass) go to
#     Tier 3; the size-as-alignment test sent both to Tier 2.
#   * a natural record the struct emitter writes as a byte blob of 16 bytes or
#     less (EnumRec: its enum member has size 0 in the metadata) goes to Tier 2.
#   * packed records go to Tier 2, including Top6, which is packed two levels
#     down and reached ccall before.
#
# Needs clang++ and libJLCS (the Tier 2 thunks).
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const CPP_ROUTING_DIR = joinpath(@__DIR__, "cpp_routing_test")

if requires("C++ record routing", :libJLCS, :clangxx)
@testset "C++ by-value record routing, live" begin
    toml_path = joinpath(CPP_ROUTING_DIR, "replibuild.toml")
    src = joinpath(CPP_ROUTING_DIR, "src", "routing.cpp")
    inc = joinpath(CPP_ROUTING_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "cpp_routing"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(CPP_ROUTING_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(inc))"]

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
    allow_function_pointers = true

    [cache]
    enabled = false
    """)

    RepliBuild.build(toml_path)
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(CPP_ROUTING_DIR, "julia", "CppRouting.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "the Julia type each record crosses as" begin
        @test occursin(r"struct CS3\n\s+c::UInt8", code)
        @test occursin(r"struct Mass\n\s+m::Cfloat", code)
        @test occursin(r"struct EnumRec\n\s+_data::NTuple\{4, UInt8\}", code)
        @test occursin(r"struct Pk5\n\s+_data::NTuple\{5, UInt8\}", code)
    end

    @testset "tier per function" begin
        tiers = Dict(Symbol(m.captures[1]) => Symbol(m.captures[2])
                     for m in eachmatch(r"\n\s+:(\w+) => :(tier\d)", code))
        want = Dict(:make_cs3 => :tier3, :cs3_sum => :tier3,
                    :make_mass => :tier3, :mass_sum => :tier3,
                    :make_enumrec => :tier2, :enumrec_sum => :tier2,
                    :make_pk5 => :tier2, :pk5_sum => :tier2,
                    :make_top6 => :tier2, :top6_sum => :tier2)
        checked = 0
        for (f, t) in sort(collect(want); by = first)
            @test (f, get(tiers, f, :missing)) == (f, t)
            checked += 1
        end
        @test checked == length(want)
    end

    @testset "every function, both directions" begin
        r = run_isolated(`$(joinpath(CPP_ROUTING_DIR, "probe_routing.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["cs3", "mass", "enumrec", "pk5", "top6"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
