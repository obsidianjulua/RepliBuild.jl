#!/usr/bin/env julia
# test/test_clean.jl — `clean(toml; deps=false)` keeps the vendored git clones.
#
# `clean` removes `.replibuild_cache/` whole, and that directory holds the git
# dependencies DependencyResolver clones under `deps/`. A Hub package's test.jl
# cleans before it rebuilds, so every run re-cloned its dependency: for
# boostmath that is minutes of the run (2026-10-25). `deps=false` empties the
# cache around `deps/` and the next build re-checks the pin markers instead.
#
# Library-free: a scratch project directory, no toolchain.

using Test
using RepliBuild

function scratch_project()
    dir = mktempdir()
    for d in ("build", "julia", ".debug",
              joinpath(".replibuild_cache", "deps", "libfoo", ".git"),
              joinpath(".replibuild_cache", "deps", "libfoo", "include"),
              joinpath(".replibuild_cache", "slices", "m"))
        mkpath(joinpath(dir, d))
    end
    write(joinpath(dir, ".replibuild_cache", "project_hash"), "abc")
    write(joinpath(dir, ".replibuild_cache", "a.ll"), "; ir")
    write(joinpath(dir, ".replibuild_cache", "deps", "libfoo", "include", "foo.h"), "int f(void);")
    write(joinpath(dir, ".replibuild_cache", "deps", "libfoo.dep"), "url\ntag\ncommit\n")
    write(joinpath(dir, "replibuild.toml"), "[project]\nname = \"foo\"\n")
    return dir
end

@testset "clean keeps or removes the vendored dependencies" begin
    @testset "default removes the whole cache" begin
        dir = scratch_project()
        RepliBuild.clean(joinpath(dir, "replibuild.toml"))
        @test !isdir(joinpath(dir, "build"))
        @test !isdir(joinpath(dir, "julia"))
        @test !isdir(joinpath(dir, ".debug"))
        @test !isdir(joinpath(dir, ".replibuild_cache"))
        @test isfile(joinpath(dir, "replibuild.toml"))
    end

    @testset "deps=false keeps deps/ and nothing else in the cache" begin
        dir = scratch_project()
        RepliBuild.clean(joinpath(dir, "replibuild.toml"); deps=false)
        @test !isdir(joinpath(dir, "build"))
        @test !isdir(joinpath(dir, "julia"))
        @test !isdir(joinpath(dir, ".debug"))
        cache = joinpath(dir, ".replibuild_cache")
        @test isdir(cache)
        @test readdir(cache) == ["deps"]
        @test isfile(joinpath(cache, "deps", "libfoo", "include", "foo.h"))
        @test isfile(joinpath(cache, "deps", "libfoo.dep"))
        @test !isfile(joinpath(cache, "project_hash"))
        @test !isdir(joinpath(cache, "slices"))
    end

    @testset "deps=false with no deps/ removes the cache directory" begin
        dir = scratch_project()
        rm(joinpath(dir, ".replibuild_cache", "deps"); recursive=true)
        RepliBuild.clean(joinpath(dir, "replibuild.toml"); deps=false)
        @test !isdir(joinpath(dir, ".replibuild_cache"))
    end

    @testset "a project with nothing to clean is a no-op" begin
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), "[project]\nname = \"foo\"\n")
        RepliBuild.clean(joinpath(dir, "replibuild.toml"); deps=false)
        @test readdir(dir) == ["replibuild.toml"]
    end
end
