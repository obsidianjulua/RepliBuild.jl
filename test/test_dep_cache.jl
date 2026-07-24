#!/usr/bin/env julia
# test/test_dep_cache.jl — the git dependency cache is version-aware
#
# Regression guard: the deps cache at .replibuild_cache/deps/<name> was keyed on
# <name> alone (a bare `isdir` check), so bumping a dependency's tag or url in the
# toml WITHOUT calling clean() silently served the previously-cloned checkout —
# a stale-serve with no fetch, no re-checkout, and no warning. A sidecar
# `<name>.resolved` marker now records the resolved url+tag; resolution
# re-checks-out on a tag change, re-clones on a url change, and reuses otherwise.
# These tests drive the real load_config → resolve_dependencies path against a
# local git upstream (needs `git` only — no C/LLVM toolchain).

using Test
using RepliBuild

const CM = RepliBuild.ConfigurationManager
const DR = RepliBuild.DependencyResolver

@testset "git dependency cache is version-aware" begin
    if Sys.which("git") === nothing
        @warn "git not found — skipping dependency cache version-awareness tests"
    else
        mktempdir() do sb
            gitrun(repo, args...) = run(`git -C $repo -c user.email=t@example.com -c user.name=tester $(collect(args))`)

            # upstream1: VERSION flips v1 -> v2 across tags v1/v2
            up1 = joinpath(sb, "upstream1"); mkpath(up1)
            run(`git -C $up1 init -q`)
            write(joinpath(up1, "VERSION"), "v1"); write(joinpath(up1, "lib.c"), "int f(){return 1;}\n")
            gitrun(up1, "add", "-A"); gitrun(up1, "commit", "-q", "-m", "v1"); gitrun(up1, "tag", "v1")
            write(joinpath(up1, "VERSION"), "v2")
            gitrun(up1, "add", "-A"); gitrun(up1, "commit", "-q", "-m", "v2"); gitrun(up1, "tag", "v2")

            # upstream2: a different repo, tag v1, VERSION = OTHER
            up2 = joinpath(sb, "upstream2"); mkpath(up2)
            run(`git -C $up2 init -q`)
            write(joinpath(up2, "VERSION"), "OTHER"); write(joinpath(up2, "lib.c"), "int g(){return 2;}\n")
            gitrun(up2, "add", "-A"); gitrun(up2, "commit", "-q", "-m", "o1"); gitrun(up2, "tag", "v1")

            proj = joinpath(sb, "proj"); mkpath(proj)
            toml_path = joinpath(proj, "replibuild.toml")
            url1 = "file://" * up1
            url2 = "file://" * up2

            write_toml(url, tag) = open(toml_path, "w") do io
                println(io, "[project]\nname = \"deptest\"\nversion = \"0.0.1\"\nroot = \"", proj, "\"\n")
                println(io, "[dependencies.fixturelib]\ntype = \"git\"\nurl = \"", url, "\"\ntag = \"", tag, "\"")
            end

            deppath = joinpath(proj, ".replibuild_cache", "deps", "fixturelib")
            verfile = joinpath(deppath, "VERSION")
            marker  = joinpath(proj, ".replibuild_cache", "deps", "fixturelib.resolved")
            readver() = strip(read(verfile, String))
            function marker_tag()
                t = ""
                isfile(marker) && for l in eachline(marker); startswith(l, "tag=") && (t = l[5:end]); end
                t
            end
            resolve() = DR.resolve_dependencies(CM.load_config(toml_path))

            # fresh clone @ tag v1
            write_toml(url1, "v1"); resolve()
            @test readver() == "v1"
            @test marker_tag() == "v1"

            # tag bump v1 -> v2 with NO clean: must re-checkout, not serve stale v1
            write_toml(url1, "v2"); resolve()
            @test readver() == "v2"
            @test marker_tag() == "v2"

            # no toml change: reused as-is
            resolve()
            @test readver() == "v2"

            # url change: re-clone from the other upstream
            write_toml(url2, "v1"); resolve()
            @test readver() == "OTHER"
            @test occursin("upstream2", read(marker, String))

            # legacy cache whose marker predates this feature: re-verify + rewrite
            rm(marker; force=true)
            write_toml(url2, "v1"); resolve()
            @test isfile(marker)
            @test marker_tag() == "v1"
        end
    end
end

println("✅ dependency cache version-awareness tests passed")
