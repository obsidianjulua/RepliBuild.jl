# =============================================================================
# Factories that share a constructor signature, live (2026-11-18)
#
# test_factory_ctor_collisions.jl (unit) checks what the generator emits from
# synthetic metadata. This builds the shapes from C++ (factory_ctor_test):
# `make_animal(int legs)` and `make_animal_doubled(int half)` both make an
# Animal, so the owning `Animal` had `Animal(::Cint)` twice and the second
# won (`Animal(Cint(4))` made 8 legs; a precompiled package did not load).
# Zebra is the same collision in the early form (a record's field names it).
# The probe makes an owning object through each factory by name, checks each
# answer, that the bare shared call is a MethodError, that a signature one
# factory has stays plain, and that the finalizers free what was made. Needs
# clang++ and libJLCS (the methods are Tier 2 thunks).
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const FACTORY_CTOR_DIR = joinpath(@__DIR__, "factory_ctor_test")
const FACTORY_CTOR_PROBE = joinpath(FACTORY_CTOR_DIR, "probe_factory_ctors.jl")

if requires("factories sharing a constructor signature, live", :libJLCS, :clangxx)
@testset "factories that share a constructor signature, live" begin
    toml_path = joinpath(FACTORY_CTOR_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "factory_ctors"
    root = "$(escape_string(FACTORY_CTOR_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(FACTORY_CTOR_DIR, "src", "factory_ctors.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(FACTORY_CTOR_DIR, "include")))"]

    [link]
    enable_lto = false

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
    code = read(wrapper, String)

    @testset "each shared signature names its factory" begin
        @test occursin("function Animal(::Base.Val{:make_animal}, legs::Cint)\n", code)
        @test occursin("function Animal(::Base.Val{:make_animal_doubled}, half::Cint)\n", code)
        @test occursin("function Animal(front::Cint, back::Cint)\n", code)
        @test !occursin(r"function Animal\((legs|half)::Cint\)", code)
        @test occursin("\nfunction Zebra(::Base.Val{:make_zebra}, stripes::Cint)\n", code)
        @test occursin("\nfunction Zebra(::Base.Val{:make_zebra_striped}, bands::Cint)\n", code)
        @test allunique(first.(RepliBuild.Wrapper._method_definitions(code)))
    end

    @testset "the wrapper loads and each factory makes its own object" begin
        names = ["wrapper_loads", "each_factory_makes_its_own_animal",
                 "a_signature_one_factory_has_stays_plain", "the_bare_shared_signature_is_a_method_error",
                 "the_early_form_makes_each_zebra", "finalizers_free_what_the_factories_made"]
        r = run_isolated(`$FACTORY_CTOR_PROBE $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
