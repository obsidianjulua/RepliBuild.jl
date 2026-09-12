# Project-level cache key — what counts as "the project changed".
#
# `compute_project_hash` gates the whole build: when it matches, `compile_project`
# prints "cache: project unchanged" and returns the existing library without
# looking at anything. So every input the compiler reads has to reach this hash,
# and anything that does NOT feed the compiler must stay out of it — a hash that
# moves on its own churns every build, one that sits still ships stale objects.
#
# The bug this pins: the include-directory walk was `readdir`, top level only.
# A header one directory down could be edited freely and the build reported
# "project unchanged" against modified sources. Not an edge case —
# `include/<lib>/<lib>.h` is the ordinary C layout, and SUNDIALS generates
# exactly `include/sundials/sundials_config.h`, so the headers most likely to
# matter were the ones least likely to be seen.
#
# Built over a synthetic project in a tempdir: no Hub package, no toolchain, and
# nothing that depends on what this box has built.

using Test
using RepliBuild

const C  = RepliBuild.Compiler
const CM = RepliBuild.ConfigurationManager

# Minimal project: one TU, one include dir, caching on.
# TOML strings treat `\U` as a unicode escape; a Windows `C:\Users\...`
# path is therefore not a valid scalar. Forward slashes are.
toml_path(p) = replace(p, '\\' => '/')

function scratch_project(dir)
    mkpath(joinpath(dir, "src"))
    mkpath(joinpath(dir, "include"))
    write(joinpath(dir, "src", "main.c"), "int f(void) { return 0; }\n")
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "probe"
    root = "$(toml_path(dir))"

    [compile]
    source_files = ["$(toml_path(joinpath(dir, "src", "main.c")))"]
    include_dirs = ["$(toml_path(joinpath(dir, "include")))"]

    [cache]
    enabled = true
    directory = ".replibuild_cache"

    [wrap]
    language = "c"
    """)
    return CM.load_config(joinpath(dir, "replibuild.toml"))
end

hashof(cfg) = C.compute_project_hash(cfg)

@testset "Project hash covers what the compiler reads" begin

    @testset "source content" begin
        mktempdir() do d
            cfg = scratch_project(d)
            h0 = hashof(cfg)
            write(joinpath(d, "src", "main.c"), "int f(void) { return 1; }\n")
            @test hashof(cfg) != h0
        end
    end

    @testset "top-level header" begin
        mktempdir() do d
            cfg = scratch_project(d)
            hdr = joinpath(d, "include", "top.h")
            write(hdr, "#define V 1\n");  h1 = hashof(cfg)
            write(hdr, "#define V 2\n");  @test hashof(cfg) != h1
        end
    end

    @testset "NESTED header — the regression" begin
        mktempdir() do d
            cfg = scratch_project(d)
            # include/lib/lib.h — the ordinary C layout the old readdir missed
            deep = joinpath(d, "include", "lib")
            mkpath(deep)
            hdr = joinpath(deep, "lib.h")

            write(hdr, "#define V 1\n"); h1 = hashof(cfg)
            write(hdr, "#define V 2\n"); h2 = hashof(cfg)
            @test h2 != h1                       # was equal: the bug

            # Deeper still, since one level of recursion would also have passed.
            deeper = joinpath(deep, "detail"); mkpath(deeper)
            write(joinpath(deeper, "impl.h"), "#define W 1\n"); h3 = hashof(cfg)
            @test h3 != h2
            write(joinpath(deeper, "impl.h"), "#define W 2\n"); @test hashof(cfg) != h3
        end
    end

    @testset "adding, renaming and removing headers" begin
        mktempdir() do d
            cfg = scratch_project(d)
            deep = joinpath(d, "include", "lib"); mkpath(deep)
            base = hashof(cfg)

            a = joinpath(deep, "a.h")
            write(a, "#define A 1\n"); h_added = hashof(cfg)
            @test h_added != base

            # A rename with identical content changes what the compiler can
            # include, so the path is hashed alongside the bytes.
            b = joinpath(deep, "b.h")
            mv(a, b); h_renamed = hashof(cfg)
            @test h_renamed != h_added

            rm(b)
            @test hashof(cfg) == base            # removal returns to baseline
        end
    end

    @testset "non-compiler inputs stay OUT of the hash" begin
        mktempdir() do d
            cfg = scratch_project(d)
            deep = joinpath(d, "include", "lib"); mkpath(deep)
            write(joinpath(deep, "real.h"), "#define R 1\n")
            base = hashof(cfg)

            # .git never feeds the compiler. Hashing it would make every fetch
            # look like a source change.
            gitdir = joinpath(d, "include", ".git"); mkpath(gitdir)
            write(joinpath(gitdir, "config.h"), "#define NOPE 1\n")
            @test hashof(cfg) == base

            # Our own outputs: hashing these makes the key depend on the build
            # it is supposed to gate.
            for sub in ("build", ".replibuild_cache")
                p = joinpath(d, "include", sub); mkpath(p)
                write(joinpath(p, "generated.h"), "#define GEN 1\n")
                @test hashof(cfg) == base
            end

            # Non-headers are not compiler inputs either.
            write(joinpath(deep, "README.md"), "not a header\n")
            @test hashof(cfg) == base
        end
    end

    @testset "the toml itself" begin
        mktempdir() do d
            cfg = scratch_project(d)
            h0 = hashof(cfg)
            toml = joinpath(d, "replibuild.toml")
            write(toml, read(toml, String) * "\n[binary]\ntype = \"shared\"\n")
            @test hashof(CM.load_config(toml)) != h0
        end
    end

    @testset "stable when nothing moves" begin
        mktempdir() do d
            cfg = scratch_project(d)
            mkpath(joinpath(d, "include", "lib", "detail"))
            write(joinpath(d, "include", "lib", "detail", "x.h"), "#define X 1\n")
            write(joinpath(d, "include", "lib", "y.h"), "#define Y 1\n")
            # Repeated calls must agree — walkdir order is sorted explicitly so
            # the key cannot depend on filesystem enumeration order.
            @test hashof(cfg) == hashof(cfg) == hashof(CM.load_config(joinpath(d, "replibuild.toml")))
        end
    end
end
