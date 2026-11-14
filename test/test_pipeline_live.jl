# The pipeline end to end on stress_test: discover → build → wrap, info and
# clean, then the chained `discover(build = true, wrap = true)`, which ends in
# `register` and vendors the build under the hash `use` looks for. Locks
# stress_test (test/suites.jl): it cleans and rebuilds the fixture, so it runs
# after the fixture job and never beside it.

using Test
using RepliBuild

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())
haskey(ENV, "REPLIBUILD_JULIA_ENV") || (ENV["REPLIBUILD_JULIA_ENV"] = "none")

const PL_DIR = joinpath(@__DIR__, "stress_test")

function pl_clean(dir::String)
    for name in ("build", "julia", ".replibuild_cache")
        p = joinpath(dir, name)
        ispath(p) && RepliBuild._rm_tree(p)
    end
end

@testset "Pipeline (discover → build → wrap)" begin
    @test isdir(PL_DIR)
    pl_clean(PL_DIR)

    toml_path = RepliBuild.discover(PL_DIR, force = true)
    @test isfile(toml_path)

    library_path = RepliBuild.build(toml_path)
    @test isfile(library_path)
    @test isfile(joinpath(PL_DIR, "julia", "compilation_metadata.json"))

    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)
    @test endswith(wrapper_path, ".jl")

    # Info / clean round-trip
    @test_nowarn RepliBuild.info(toml_path)
    RepliBuild.clean(toml_path)
    @test !isdir(joinpath(PL_DIR, "build"))
    @test !isdir(joinpath(PL_DIR, "julia"))

    # Chained pipeline
    toml_path = RepliBuild.discover(PL_DIR, force = true, build = true, wrap = true)
    @test isfile(toml_path)
    julia_dir = joinpath(PL_DIR, "julia")
    @test any(endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll") for f in readdir(julia_dir))
    @test any(endswith(f, ".jl") for f in readdir(julia_dir))

    # It ends in `register`: the build made in the project is vendored (to this
    # run's temp REPLIBUILD_HOME) under the hash `use` will look for.
    cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
    entry = RepliBuild.PackageRegistry._load_index().entries[cfg.project.name]
    @test entry.toml_path == abspath(toml_path)
    @test isfile(joinpath(RepliBuild.PackageRegistry._cached_build_dir(entry.build_hash),
                          RepliBuild.ConfigurationManager.get_module_name(cfg) * ".jl"))
end
