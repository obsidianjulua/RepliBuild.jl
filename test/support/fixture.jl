#!/usr/bin/env julia
# test/support/fixture.jl <name> — build and verify one discover-driven fixture
# project (test/suites.jl's FIXTURES): the job `fixture:<name>` of the dev suite.
#
# discover(force = true) writes the fixture's TOML afresh (they are gitignored:
# a committed one would carry machine paths), the curated config discovery
# cannot derive is seeded after that and before the build (what regressed
# stl_test for six weeks when discover(force) destroyed [types].templates,
# 2026-07-17), then build, wrap, and the fixture's verify.jl in a child of its
# own (module names collide between fixtures). Runs as a @testset, so a failure
# is a non-zero exit for run.jl; in-process, run.jl calls `run_fixture`.

using Test
using TOML
using RepliBuild

isdefined(@__MODULE__, :FIXTURES) || include(joinpath(@__DIR__, "..", "suites.jl"))

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())
haskey(ENV, "REPLIBUILD_JULIA_ENV") || (ENV["REPLIBUILD_JULIA_ENV"] = "none")

function fixture_named(name::AbstractString)
    i = findfirst(f -> f.name == name, FIXTURES)
    i === nothing && error("fixture.jl: no fixture `$name` in test/suites.jl (" *
                           join([f.name for f in FIXTURES], ", ") * ")")
    return FIXTURES[i]
end

function clean_fixture_dir(dir::String)
    for name in ("build", "julia", ".replibuild_cache")
        p = joinpath(dir, name)
        # `_rm_tree`, not a bare rm: on Windows a file with an open handle cannot be
        # unlinked and the directory reports ENOTEMPTY; it retries with a backoff.
        ispath(p) && RepliBuild._rm_tree(p)
    end
end

function apply_curated_config(toml_path::String, curated::AbstractDict)
    isempty(curated) && return
    doc = TOML.parsefile(toml_path)
    for (sec, kv) in curated, (k, v) in kv
        get!(doc, sec, Dict{String,Any}())[k] = v
    end
    open(toml_path, "w") do io
        TOML.print(io, doc)
    end
end

"Build, wrap and verify one fixture, as a testset named after it."
function run_fixture(name::AbstractString)
    fx = fixture_named(name)
    @testset "$(fx.label)" begin
        dir = joinpath(TEST_DIR, fx.name)
        @test isdir(dir)
        clean_fixture_dir(dir)
        toml = RepliBuild.discover(dir, force = true)
        @test isfile(toml)
        apply_curated_config(toml, fx.curated)
        RepliBuild.build(toml)
        RepliBuild.wrap(toml)
        @test isdir(joinpath(dir, "julia"))

        verify = joinpath(dir, "verify.jl")
        @test isfile(verify)
        # The verify script in a child: a crash takes the child, not this job's
        # report, and two fixtures' modules never share a process.
        cmd = `$(Base.julia_cmd()) --project=$(dirname(TEST_DIR)) $verify`
        @test success(run(ignorestatus(cmd)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("fixture.jl: one argument, a fixture name from test/suites.jl")
    run_fixture(ARGS[1])
end
