# =============================================================================
# Class-nested types under their qualified names, live (2026-11-15)
#
# test_nested_type_aliases.jl (unit) checks which records get an alias and
# the emitted text. This builds test/nested_type_test (types declared inside
# a class, as Bullet's btCollisionWorld::ClosestRayResultCallback is), wraps
# it, and uses the types under the names their functions carry, from a child.
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const NESTED_TYPE_DIR = joinpath(@__DIR__, "nested_type_test")

if requires("class-nested type aliases, live", :libJLCS, :clangxx)
@testset "Class-nested type aliases, live" begin
    toml_path = joinpath(NESTED_TYPE_DIR, "replibuild.toml")
    src = joinpath(NESTED_TYPE_DIR, "src", "nested_type.cpp")
    inc = joinpath(NESTED_TYPE_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "nestedtype"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(NESTED_TYPE_DIR))"

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
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(NESTED_TYPE_DIR, "julia", "Nestedtype.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "the metadata's scope is the enclosing class" begin
        md = RepliBuild._read_json(joinpath(NESTED_TYPE_DIR, "julia", "compilation_metadata.json"))
        sd = md["struct_definitions"]
        @test sd["RayResult"]["scope"] == "World"
        @test sd["Info"]["scope"] == "World::Callback"
        @test sd["Lid"]["scope"] == "nt::Box"
        @test sd["Plain"]["scope"] == "nt"
    end

    @testset "the wrapper binds each class-nested type under its functions' prefix" begin
        aliases = [m.captures[1] => m.captures[2] for m in eachmatch(r"^const (\w+) = (\w+)$"m,
                   something(match(r"# ── Class-nested types.*?\n\n"s, code), (match = "",)).match)]
        @test aliases == ["World_Callback" => "Callback", "World_Callback_Info" => "Info",
                          "World_RayResult" => "RayResult", "nt_Box_Lid" => "Lid"]
        # The bare names are what they were.
        @test occursin("\nstruct RayResult\n", code) && occursin("\nstruct Mark\n", code)
        # The functions carry the same prefix the alias has.
        @test occursin("\nfunction World_RayResult_hit(this::Any)", code)
        @test occursin(r"\nfunction World_Callback_Info_Info\(this::Any, code::Integer\)", code)
        # `World_Mark` is the free function; no alias and no layout key took it.
        @test !occursin("const World_Mark =", code)
        @test !occursin(":World_Mark =>", code)
        @test occursin("    :World_RayResult => 8,\n", code)
    end

    @testset "used from a child under the qualified names" begin
        r = run_isolated(`$(joinpath(NESTED_TYPE_DIR, "probe_nested_type.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["alias_is_the_type", "layout_by_both_names", "one_name_end_to_end", "taken_name_not_rebound"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
