#!/usr/bin/env julia
# test/test_clean.jl — what `clean` does with a project's downloaded sources.
#
# `.replibuild_cache/deps/` holds what DependencyResolver downloaded: git
# checkouts and unpacked archives. Deleting it makes the next build download
# the same sources again, which for boostmath is minutes (2026-10-25), so
# `deps=false` was added to keep it.
#
# 2026-11-23 (John: "I dont like git pulls unless sources are new", "clean
# should prompt to delete sources"): keeping them is now what `clean(toml)`
# does. Deleting them takes `deps=true`, or a `y` to the question `clean` asks
# at an interactive prompt. A script is never asked and never loses them, and
# `build(toml; clean=true)` keeps them always.
#
# Library-free: a scratch project directory, no toolchain. Nothing here is an
# interactive session, so the question itself is driven through
# `_clean_deletes_sources` with its input and output given.

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

# `clean` with its output captured: what it printed, and nothing on the terminal.
function quiet_clean(args...; kwargs...)
    out = Pipe()
    redirect_stdout(out) do
        RepliBuild.clean(args...; kwargs...)
    end
    close(out.in)
    return read(out, String)
end

function kept_sources(dir)
    cache = joinpath(dir, ".replibuild_cache")
    return isdir(cache) && readdir(cache) == ["deps"] &&
           isfile(joinpath(cache, "deps", "libfoo", "include", "foo.h")) &&
           isfile(joinpath(cache, "deps", "libfoo.dep"))
end

@testset "clean and a project's downloaded sources" begin
    @testset "clean(toml) in a script keeps them, and says how to delete them" begin
        @test !isinteractive()                      # the suite is a script: nobody can be asked
        dir = scratch_project()
        printed = quiet_clean(joinpath(dir, "replibuild.toml"))
        @test !isdir(joinpath(dir, "build"))
        @test !isdir(joinpath(dir, "julia"))
        @test !isdir(joinpath(dir, ".debug"))
        @test kept_sources(dir)
        @test !isfile(joinpath(dir, ".replibuild_cache", "project_hash"))
        @test !isdir(joinpath(dir, ".replibuild_cache", "slices"))
        @test isfile(joinpath(dir, "replibuild.toml"))
        @test occursin("downloaded sources kept", printed) && occursin("deps=true", printed)
    end

    @testset "deps=true deletes them, and asks nothing" begin
        dir = scratch_project()
        printed = quiet_clean(joinpath(dir, "replibuild.toml"); deps=true)
        @test !isdir(joinpath(dir, "build")) && !isdir(joinpath(dir, "julia")) && !isdir(joinpath(dir, ".debug"))
        @test !isdir(joinpath(dir, ".replibuild_cache"))
        @test isfile(joinpath(dir, "replibuild.toml"))
        @test !occursin("downloaded sources", printed)
    end

    @testset "deps=false keeps deps/ and nothing else in the cache" begin
        dir = scratch_project()
        printed = quiet_clean(joinpath(dir, "replibuild.toml"); deps=false)
        @test !isdir(joinpath(dir, "build"))
        @test !isdir(joinpath(dir, "julia"))
        @test !isdir(joinpath(dir, ".debug"))
        @test kept_sources(dir)
        @test !isfile(joinpath(dir, ".replibuild_cache", "project_hash"))
        @test !isdir(joinpath(dir, ".replibuild_cache", "slices"))
        @test !occursin("downloaded sources", printed)             # told what to do: nothing to say
    end

    @testset "with nothing downloaded the whole cache goes, whatever `deps` is" begin
        for deps in (nothing, false, true)
            dir = scratch_project()
            rm(joinpath(dir, ".replibuild_cache", "deps"); recursive=true)
            printed = deps === nothing ? quiet_clean(joinpath(dir, "replibuild.toml")) :
                      quiet_clean(joinpath(dir, "replibuild.toml"); deps=deps)
            @test (deps, isdir(joinpath(dir, ".replibuild_cache"))) == (deps, false)
            @test !occursin("downloaded sources", printed)         # nothing to ask about
        end
        # An empty deps/ folder is nothing downloaded either.
        dir = scratch_project()
        rm(joinpath(dir, ".replibuild_cache", "deps"); recursive=true)
        mkpath(joinpath(dir, ".replibuild_cache", "deps"))
        quiet_clean(joinpath(dir, "replibuild.toml"))
        @test !isdir(joinpath(dir, ".replibuild_cache"))
    end

    @testset "a project with nothing to clean is a no-op" begin
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), "[project]\nname = \"foo\"\n")
        quiet_clean(joinpath(dir, "replibuild.toml"))
        @test readdir(dir) == ["replibuild.toml"]
    end

    @testset "the question: only `y` deletes" begin
        decide = RepliBuild._clean_deletes_sources
        deps_dir = joinpath(scratch_project(), ".replibuild_cache", "deps")
        ask(answer) = (out = IOBuffer();
                       (decide(nothing, deps_dir; input = IOBuffer(answer), output = out, can_ask = true), String(take!(out))))

        (delete, said) = ask("y\n")
        @test delete
        # It names what it would delete, how much, and what deleting costs.
        @test occursin("libfoo", said) && occursin(deps_dir, said)
        @test occursin("Delete them too? The next build would download them again. [y/N]", said)
        @test !occursin("libfoo.dep", said)                        # a marker file is not a source
        @test first(ask("YES\n")) && first(ask("  Y  \n"))
        # Anything else keeps them: no, an empty line, a typo, end of input.
        for answer in ("n\n", "\n", "no\n", "yy\n", "delete\n", "")
            (delete, said) = ask(answer)
            @test (answer, delete) == (answer, false)
            @test occursin("downloaded sources kept", said)
        end

        # `deps` given is the answer: nothing is read, nothing is printed.
        for given in (true, false)
            out = IOBuffer()
            @test decide(given, deps_dir; input = IOBuffer("y\n"), output = out, can_ask = true) == given
            @test isempty(take!(out))
        end
        # Nobody to ask: kept, with the line that says how to delete, and the input untouched.
        input = IOBuffer("y\n"); out = IOBuffer()
        @test !decide(nothing, deps_dir; input = input, output = out, can_ask = false)
        @test position(input) == 0
        @test occursin("`clean(toml; deps=true)` deletes them", String(take!(out)))
        # Nothing downloaded: no question, and the cache may go whole.
        out = IOBuffer()
        @test decide(nothing, joinpath(mktempdir(), "absent"); input = IOBuffer("n\n"), output = out, can_ask = true)
        @test isempty(take!(out))

        # A person can be asked only in an interactive session on a terminal.
        @test !RepliBuild._clean_can_ask(IOBuffer(), IOBuffer())
        @test !RepliBuild._clean_can_ask(stdin, IOBuffer())
    end

    @testset "build(toml; clean=true) keeps the downloaded sources" begin
        # Read off the source: the clean inside `build` never deletes sources and
        # never asks. (A build needs a compiler; test_dep_cache.jl and the dev
        # suite's graph test run the real thing.)
        src = read(joinpath(pkgdir(RepliBuild), "src", "RepliBuild.jl"), String)
        body = src[findfirst("function build(toml_path::String", src).start:findfirst("function wrap(", src).start]
        @test occursin("clean_internal(project_dir; deps=false)", body)
        @test !occursin("clean_internal(project_dir)", body)
        @test !occursin("_clean_deletes_sources", body)
    end
end
