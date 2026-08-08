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
            # Signing is disabled for the same reason identity is pinned: this
            # must not depend on the developer's global git config. With
            # `tag.gpgsign=true` set globally, a lightweight `git tag v1` below
            # becomes a SIGNED tag, which requires a message and dies with
            # "fatal: no tag message?" — a failure naming nothing about
            # dependency caching.
            gitrun(repo, args...) = run(`git -C $repo -c user.email=t@example.com -c user.name=tester -c commit.gpgsign=false -c tag.gpgsign=false $(collect(args))`)

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

@testset "git dependency commit pinning" begin
    if Sys.which("git") === nothing
        @warn "git not found — skipping dependency commit-pinning tests"
    else
        mktempdir() do sb
            # Signing is disabled for the same reason identity is pinned: this
            # must not depend on the developer's global git config. With
            # `tag.gpgsign=true` set globally, a lightweight `git tag v1` below
            # becomes a SIGNED tag, which requires a message and dies with
            # "fatal: no tag message?" — a failure naming nothing about
            # dependency caching.
            gitrun(repo, args...) = run(`git -C $repo -c user.email=t@example.com -c user.name=tester -c commit.gpgsign=false -c tag.gpgsign=false $(collect(args))`)
            gitout(repo, args...) = strip(read(`git -C $repo $(collect(args))`, String))

            # upstream: tag v1 -> commit A (GOOD). Later force-moved to commit B (EVIL).
            up = joinpath(sb, "upstream"); mkpath(up)
            run(`git -C $up init -q`)
            write(joinpath(up, "VERSION"), "GOOD"); write(joinpath(up, "lib.c"), "int f(){return 1;}\n")
            gitrun(up, "add", "-A"); gitrun(up, "commit", "-q", "-m", "good"); gitrun(up, "tag", "v1")
            sha_good = lowercase(gitout(up, "rev-parse", "v1^{commit}"))

            proj = joinpath(sb, "proj"); mkpath(proj)
            toml_path = joinpath(proj, "replibuild.toml")
            url = "file://" * up

            write_toml(tag, commit="") = open(toml_path, "w") do io
                println(io, "[project]\nname = \"pintest\"\nversion = \"0.0.1\"\nroot = \"", proj, "\"\n")
                print(io, "[dependencies.fixturelib]\ntype = \"git\"\nurl = \"", url, "\"\ntag = \"", tag, "\"")
                println(io, isempty(commit) ? "" : "\ncommit = \"" * commit * "\"")
            end

            depsdir = joinpath(proj, ".replibuild_cache", "deps")
            deppath = joinpath(depsdir, "fixturelib")
            marker  = joinpath(depsdir, "fixturelib.resolved")
            readver() = strip(read(joinpath(deppath, "VERSION"), String))
            function marker_commit()
                c = ""
                isfile(marker) && for l in eachline(marker); startswith(l, "commit=") && (c = l[8:end]); end
                c
            end
            resolve() = DR.resolve_dependencies(CM.load_config(toml_path))
            # Hub `test.jl` rebuilds cold via clean(), which deletes the clone AND the
            # marker — the pin is the only layer that survives it, so exercise that path.
            wipe() = rm(depsdir; recursive=true, force=true)

            # --- the marker records the resolved object name -------------------
            write_toml("v1"); resolve()
            @test readver() == "GOOD"
            @test marker_commit() == sha_good
            @test occursin(r"^[0-9a-f]{40}$", marker_commit())

            # --- a matching pin resolves normally ------------------------------
            wipe(); write_toml("v1", sha_good); resolve()
            @test readver() == "GOOD"
            @test marker_commit() == sha_good

            # --- THE ATTACK: upstream force-moves the tag to different content --
            write(joinpath(up, "VERSION"), "EVIL"); write(joinpath(up, "lib.c"), "int f(){system(\"pwn\");return 1;}\n")
            gitrun(up, "add", "-A"); gitrun(up, "commit", "-q", "-m", "evil")
            gitrun(up, "tag", "-f", "v1")
            sha_evil = lowercase(gitout(up, "rev-parse", "v1^{commit}"))
            @test sha_evil != sha_good

            # Unpinned + cold: the moved tag is fetched with no signal whatsoever.
            # This is the pre-fix behaviour, asserted so the pin's value is explicit.
            wipe(); write_toml("v1"); resolve()
            @test readver() == "EVIL"
            @test marker_commit() == sha_evil

            # Pinned + cold: same fetch, but the build is REFUSED.
            wipe(); write_toml("v1", sha_good)
            @test_throws ErrorException resolve()
            # And it must not leave a marker claiming the content was resolved.
            @test !isfile(marker) || marker_commit() != sha_good

            # --- a pin the recipe updates deliberately is not an error ----------
            wipe(); write_toml("v1", sha_evil); resolve()
            @test readver() == "EVIL"
            @test marker_commit() == sha_evil

            # --- local checkout drift is detected and re-resolved ---------------
            # Roll the cached clone back to the good commit behind the resolver's back.
            gitrun(deppath, "checkout", "-q", "--force", sha_good)
            @test strip(read(joinpath(deppath, "VERSION"), String)) == "GOOD"
            resolve()   # marker says EVIL, HEAD says GOOD -> re-resolve to the declared pin
            @test readver() == "EVIL"
            @test marker_commit() == sha_evil

            # --- malformed pins are rejected at PARSE time ----------------------
            # An abbreviated sha must not silently read as "no pin".
            wipe(); write_toml("v1", sha_good[1:12])
            @test_throws ErrorException CM.load_config(toml_path)
            write_toml("v1", "not-a-sha")
            @test_throws ErrorException CM.load_config(toml_path)
        end
    end
end

println("✅ dependency cache version-awareness tests passed")
println("✅ dependency commit-pinning tests passed")
