#!/usr/bin/env julia
# test/test_dep_archive.jl — archive dependencies: a tarball pinned by SHA-256
#
# A library that ships as a tarball (NASA's CDF, 2026-11-07) has no git URL to
# pin, and the two ways around it were both wrong: vendoring the tree into the
# Hub, or a `type = "local"` dependency fed by a hand-written fetch script. The
# resolver now takes `type = "archive"` with `url` and `sha256`, fetches through
# Pkg's download-verify-unpack engine into the deps cache, records the pin in a
# marker, and treats the unpacked tree like a clone: `subdir` (or the single
# top-level directory) is the root, and `exclude` applies.
#
# No toolchain and no network: the tarball is made here with
# `Pkg.PlatformEngines.package` and reached through a `file://` URL (a plain
# path in the recipe is spelled into one by the resolver on either host).
#
# Negative checks: drop the sha256 line from the recipe and load refuses it;
# change one hex digit and resolution refuses the file; neuter `_dep_excluded`
# to return false and the exclude assertions fail.

using Test
using RepliBuild
using Pkg
using SHA

const CM = RepliBuild.ConfigurationManager
const DR = RepliBuild.DependencyResolver

# A staged tree whose tarball unpacks to ONE top-level directory, as upstream
# tarballs do: stage/lib-1.0/{src,include,tests,tools}.
function _stage_upstream(stage)
    up = joinpath(stage, "lib-1.0")
    mkpath(joinpath(up, "src")); mkpath(joinpath(up, "include")); mkpath(joinpath(up, "tests")); mkpath(joinpath(up, "tools"))
    write(joinpath(up, "src", "a.c"), "int a(void){return 1;}\n")
    write(joinpath(up, "src", "b.c"), "int b(void){return 2;}\n")
    write(joinpath(up, "include", "lib.h"), "int a(void); int b(void);\n")
    write(joinpath(up, "tests", "t.c"), "int t(void){return 0;}\n")
    write(joinpath(up, "tools", "cli.c"), "int main(void){return 0;}\n")
    return up
end

_toml(sb, url, sha; subdir="", exclude="[]", extra="") = """
[project]
name = "archdep"
root = "."

[compile]
source_files = []

[dependencies.lib]
type = "archive"
url = "$(escape_string(url))"
$(isempty(sha) ? "" : "sha256 = \"$sha\"")
$(isempty(subdir) ? "" : "subdir = \"$subdir\"")
exclude = $exclude
$extra
"""

@testset "archive dependencies" begin
    mktempdir() do sb
        stage = joinpath(sb, "stage"); mkpath(stage)
        _stage_upstream(stage)
        tarball = joinpath(sb, "lib-1.0.tar.gz")
        Pkg.PlatformEngines.package(stage, tarball)
        @test isfile(tarball)
        sha = bytes2hex(open(sha256, tarball))
        proj = joinpath(sb, "proj"); mkpath(proj)
        toml_path = joinpath(proj, "replibuild.toml")
        deps_cache = joinpath(proj, ".replibuild_cache", "deps")
        dep_dir = joinpath(deps_cache, "lib")

        @testset "the recipe must pin the hash, and only an archive takes archive keys" begin
            write(toml_path, _toml(sb, tarball, ""))
            @test_throws ErrorException CM.load_config(toml_path)
            write(toml_path, _toml(sb, tarball, "abc123"))
            @test_throws ErrorException CM.load_config(toml_path)
            write(toml_path, _toml(sb, tarball, sha; subdir="../escape"))
            @test_throws ErrorException CM.load_config(toml_path)
            write(toml_path, """
            [project]
            name = "archdep"
            root = "."
            [dependencies.lib]
            type = "local"
            path = "vendor"
            sha256 = "$sha"
            """)
            @test_throws ErrorException CM.load_config(toml_path)
            write(toml_path, _toml(sb, tarball, sha))
            cfg = CM.load_config(toml_path)
            dep = cfg.dependencies.items["lib"]
            @test dep.type == "archive" && dep.sha256 == sha && dep.subdir == ""
        end

        @testset "fetched, verified, unpacked, excluded" begin
            write(toml_path, _toml(sb, tarball, sha; exclude="[\"tools/\"]"))
            cfg = DR.resolve_dependencies(CM.load_config(toml_path))
            root = joinpath(dep_dir, "lib-1.0")                       # the single top-level directory
            @test isdir(root)
            srcs = Set(replace(relpath(s, root), "\\" => "/") for s in cfg.compile.source_files)
            @test srcs == Set(["src/a.c", "src/b.c"])                 # tests/ skipped by the walker, tools/ by exclude
            @test root in cfg.compile.include_dirs
            @test joinpath(root, "include") in cfg.compile.include_dirs
            @test joinpath(root, "src") in cfg.compile.include_dirs
            marker = joinpath(deps_cache, "lib.resolved")
            @test isfile(marker)
            @test DR._read_archive_marker(marker) == (tarball, sha)
        end

        @testset "a cache hit is reused; a changed pin or url re-fetches" begin
            sentinel = joinpath(dep_dir, "lib-1.0", "SENTINEL")
            write(sentinel, "still here")
            DR.resolve_dependencies(CM.load_config(toml_path))
            @test isfile(sentinel)                                      # same url, same sha: untouched
            # the same content from another path: a url change re-fetches
            copy = joinpath(sb, "mirror", "lib-1.0.tar.gz"); mkpath(dirname(copy)); cp(tarball, copy)
            write(toml_path, _toml(sb, copy, sha))
            cfg = DR.resolve_dependencies(CM.load_config(toml_path))
            @test !isfile(sentinel)
            @test any(endswith("a.c"), cfg.compile.source_files)
            @test DR._read_archive_marker(joinpath(deps_cache, "lib.resolved"))[1] == copy
        end

        @testset "a hash mismatch is a hard error and leaves nothing behind" begin
            wrong = (sha[1] == 'f' ? "0" : "f") * sha[2:end]
            write(toml_path, _toml(sb, tarball, wrong))
            err = try DR.resolve_dependencies(CM.load_config(toml_path)); nothing catch e; e end
            @test err isa ErrorException
            @test occursin("sha256", sprint(showerror, err))
            @test !isdir(dep_dir)
        end

        @testset "subdir names the root inside the archive" begin
            write(toml_path, _toml(sb, tarball, sha; subdir="lib-1.0/src"))
            cfg = DR.resolve_dependencies(CM.load_config(toml_path))
            root = joinpath(dep_dir, "lib-1.0", "src")
            srcs = Set(relpath(s, root) for s in cfg.compile.source_files)
            @test srcs == Set(["a.c", "b.c"])
            @test root in cfg.compile.include_dirs
            @test !(joinpath(dep_dir, "lib-1.0", "include") in cfg.compile.include_dirs)
            # a subdir the archive does not have is a warning and contributes nothing
            write(toml_path, _toml(sb, tarball, sha; subdir="lib-1.0/nope"))
            cfg = @test_logs (:warn, r"not a directory") match_mode=:any DR.resolve_dependencies(CM.load_config(toml_path))
            @test isempty(cfg.compile.source_files)
        end

        @testset "a file:// url and a plain path mean the same file" begin
            url = DR._archive_url(tarball, proj)
            @test startswith(url, "file:///")
            @test !occursin('\\', url)
            @test DR._archive_url("https://example.org/x.tar.gz", proj) == "https://example.org/x.tar.gz"
            write(toml_path, _toml(sb, url, sha))
            cfg = DR.resolve_dependencies(CM.load_config(toml_path))
            @test any(endswith("b.c"), cfg.compile.source_files)
        end
    end
end
