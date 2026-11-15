# =============================================================================
# Records with no data member of their own, live (2026-11-14)
#
# test_memberless_records.jl (unit) checks the parser, the verdicts and the
# emitted text from a vendored dump. This builds the same source
# (memberless_test) and crosses each record from a child process: a derived
# class whose bytes all come from its base goes in and comes back by value, an
# empty functor is dropped so the integers after it arrive, Bullet's shape (a
# polymorphic class adding only methods) is sized by `struct_size` and built by
# its constructor on that storage, and a class with a virtual base is sized
# without being laid out from it. Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const MEMBERLESS_DIR = joinpath(@__DIR__, "memberless_test")

if requires("memberless records, live", :libJLCS, :clangxx)
@testset "Memberless records cross by value, live" begin
    toml_path = joinpath(MEMBERLESS_DIR, "replibuild.toml")
    src = joinpath(MEMBERLESS_DIR, "src", "memberless.cpp")
    inc = joinpath(MEMBERLESS_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "memberless"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(MEMBERLESS_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-O2"]
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

    [cache]
    enabled = false
    """)

    RepliBuild.build(toml_path)
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(MEMBERLESS_DIR, "julia", "Memberless.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "the metadata has an entry for each" begin
        md = RepliBuild._read_json(joinpath(MEMBERLESS_DIR, "julia", "compilation_metadata.json"))
        sd = md["struct_definitions"]
        checked = 0
        for (k, bases) in (("Derived", ["Quad"]), ("P", String[]), ("PolyKid", ["Poly"]), ("VB", ["Poly", "Quad"]))
            @test (k, haskey(sd, k)) == (k, true)
            haskey(sd, k) || continue
            @test (k, sd[k]["members"]) == (k, Any[])
            @test (k, [b["type"] for b in get(sd[k], "base_classes", Any[])]) == (k, bases)
            checked += 1
        end
        @test checked == 4
    end

    @testset "the Julia type each record gets" begin
        @test occursin("\nstruct Derived\n    m::NTuple{4, Cfloat}\nend\n", code)
        @test occursin("\nstruct P\n    _data::NTuple{1, UInt8}\nend\n", code)
        @test occursin(r"\nstruct PolyKid\n    _vptr_Poly::Ptr\{Cvoid\}\n    id::Cint\n", code)
        @test occursin(r"\nstruct VB\n    _vptr_Poly::Ptr\{Cvoid\}\n    id::Cint\n    _pad_tail::NTuple\{20, UInt8\}\nend\n", code)
    end

    @testset "every crossing, from a child" begin
        r = run_isolated(`$(joinpath(MEMBERLESS_DIR, "probe_memberless.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["layout_tables_agree", "derived_by_value", "empty_functor_dropped",
                 "polykid_sized_and_constructed", "virtual_base_sized_not_flattened"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
