# Ingest mode (BYOB) end to end: build c_test from source, then point `ingest`
# at the resulting library and confirm the wrapper generates without the compile
# pipeline, and that `build = true, wrap = true` registers a vendored build as
# discover does. Locks c_test (test/suites.jl): it rebuilds the fixture.

using Test
using RepliBuild

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())
haskey(ENV, "REPLIBUILD_JULIA_ENV") || (ENV["REPLIBUILD_JULIA_ENV"] = "none")

const IN_DIR = joinpath(@__DIR__, "c_test")

function in_clean(dir::String)
    for name in ("build", "julia", ".replibuild_cache")
        p = joinpath(dir, name)
        ispath(p) && RepliBuild._rm_tree(p)
    end
end

@testset "Ingest mode (BYOB)" begin
    @test isdir(IN_DIR)

    # Source-build c_test to produce libc_test.so with DWARF. The fixture job
    # before this one wrote its TOML; a run of this file alone discovers it.
    in_clean(IN_DIR)
    src_toml = joinpath(IN_DIR, "replibuild.toml")
    isfile(src_toml) || RepliBuild.discover(IN_DIR, force = true)
    @test isfile(src_toml)
    src_lib = RepliBuild.build(src_toml)
    @test isfile(src_lib)

    # Ingest that .so into a brand-new project that has no source whatsoever.
    mktempdir() do ingest_dir
        toml = RepliBuild.ingest(src_lib, headers = [joinpath(IN_DIR, "include")], name = "c_ingest_e2e",
                                 project_dir = ingest_dir, language = :c, register = false)
        @test isfile(toml)

        ingested_lib = RepliBuild.build(toml)
        @test isfile(ingested_lib)
        @test isfile(joinpath(ingest_dir, "julia", "compilation_metadata.json"))

        wrapper = RepliBuild.wrap(toml)
        @test isfile(wrapper)
        @test endswith(wrapper, ".jl")
    end

    # `build = true, wrap = true` registers after wrapping, as discover does: the
    # entry carries a vendored build.
    mktempdir() do tmp
        withenv("REPLIBUILD_HOME" => joinpath(tmp, "rb_home"), "REPLIBUILD_JULIA_ENV" => "none") do
            toml = RepliBuild.ingest(src_lib, headers = [joinpath(IN_DIR, "include")], name = "c_ingest_reg",
                                     project_dir = joinpath(tmp, "proj"), language = :c, build = true, wrap = true)
            entry = RepliBuild.PackageRegistry._load_index().entries["c_ingest_reg"]
            @test entry.toml_path == abspath(toml)
            @test !isempty(entry.build_hash)
            @test isfile(joinpath(tmp, "rb_home", "builds", entry.build_hash, entry.module_name * ".jl"))
            @test isempty(entry.julia_env)
        end
    end
end
