# =============================================================================
# A record's field may name a factory class, live (2026-11-18)
#
# test_factory_field_order.jl (unit) checks what the generator emits from
# synthetic metadata. This builds the same shapes from C++
# (factory_field_test): records holding an `Animal*`, an `Animal**`, an array
# of `Animal*` and a `Bird*`, where Animal and Bird have factories and
# deleters, so the wrapper makes them owning Julia types. Before the fix the
# wrapper failed to load (`UndefVarError: Animal`): the owning type was
# defined after every record. The probe loads it, builds each record around
# objects the owning types made, passes it to the library, reads the pointer
# back, and calls a factory that takes the record holding its class. Needs
# clang++ and libJLCS (the methods are Tier 2 thunks).
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const FACTORY_FIELD_DIR = joinpath(@__DIR__, "factory_field_test")
const FACTORY_FIELD_PROBE = joinpath(FACTORY_FIELD_DIR, "probe_factory_fields.jl")

if requires("factory classes named by record fields, live", :libJLCS, :clangxx)
@testset "a record's field may name a factory class, live" begin
    toml_path = joinpath(FACTORY_FIELD_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "factory_fields"
    root = "$(escape_string(FACTORY_FIELD_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(FACTORY_FIELD_DIR, "src", "factory_fields.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(FACTORY_FIELD_DIR, "include")))"]

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

    @testset "the classes' types precede the records" begin
        at(s) = (r = findfirst(s, code); r === nothing ? 0 : first(r))
        records = [at("\nstruct $r\n") for r in ("Cage", "Den", "Ark", "Aviary")]
        @test all(>(0), records)
        for cls in ("Animal", "Bird")
            ty = at("\nmutable struct $cls\n    handle::Ptr{Cvoid}\n")
            @test (cls, 0 < ty < minimum(records)) == (cls, true)
            @test (cls, at("\nfunction $cls(") > maximum(records)) == (cls, true)
        end
        @test occursin("    pet::Ptr{Animal}\n", code)
        @test occursin("    pack::Ptr{Ptr{Animal}}\n", code)
        @test occursin("    pets::NTuple{3, Ptr{Animal}}\n", code)
        @test occursin("    bird::Ptr{Bird}\n", code)
        # The cycle's constructor: a factory of Animal taking the Cage.
        @test occursin("\nfunction Animal(c::Cage)\n", code)
    end

    @testset "the wrapper loads and calls through the records" begin
        names = ["wrapper_loads", "fields_keep_the_class_type", "cage_holds_an_animal",
                 "den_holds_pointers_to_animals", "ark_holds_an_array_of_animals",
                 "a_factory_takes_the_record_that_holds_its_class",
                 "a_second_factory_class_and_its_record", "finalizers_free_what_the_factories_made"]
        r = run_isolated(`$FACTORY_FIELD_PROBE $wrapper`)
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
