#!/usr/bin/env julia
# test/test_compile_visibility.jl — the `[compile] visibility` key
#
# `visibility = "hidden"` is the declared form of `-fvisibility=hidden`, and it is
# not an ordinary flag: it decides what the generated wrapper's API IS. Under it,
# every definition the library did not annotate `visibility("default")` stops
# reaching `.dynsym`, so the wrap surface narrows to the set upstream declared
# public — exact and free for a library that annotates, total erasure for one that
# does not. `VisibilityProbe` says which kind a package is; `verify_wrap_surface`
# refuses the erasure; this key is how the answer gets recorded.
#
# Four properties, each of which fails silently if it regresses:
#
#   * CONSUMED VIA get_compile_flags, because that is what
#     compute_compile_fingerprint hashes. Fold the flag in further downstream and
#     flipping the key leaves the per-file IR cache valid — the next build serves
#     IR compiled under the other visibility and the wrapper's API belongs to the
#     previous setting, with no error anywhere.
#   * SERIALIZED. Dropping `"hidden"` on a round trip silently restores the
#     library's internals to the wrapper's API.
#   * PRESERVED across `discover(force=true)`. Nothing in a source tree says
#     whether a library annotates its exports, so this is user intent — the
#     `[types].templates` class that killed stl_test for six weeks.
#   * VALIDATED. A typo'd value, or a key that contradicts a raw `-fvisibility=`
#     in `flags`, must refuse rather than pick one.
#
# No toolchain: parse, accessor, hash, serialize, preserve.

using Test
using TOML
using RepliBuild

const CM   = RepliBuild.ConfigurationManager
const DISC = RepliBuild.Discovery
const COMP = RepliBuild.Compiler

_cfg(dir, name, compile_body) = begin
    p = joinpath(dir, name)
    write(p, "[project]\nname = \"$(splitext(name)[1])\"\nroot = \".\"\n\n[compile]\n$compile_body")
    CM.load_config(p)
end

@testset "[compile] visibility" begin

    mktempdir() do dir

        @testset "parse + accessor" begin
            c = _cfg(dir, "a.toml", "flags = [\"-O2\"]\n")
            @test c.compile.visibility === :default
            @test CM.get_compile_flags(c) == ["-O2"]

            h = _cfg(dir, "b.toml", "flags = [\"-O2\"]\nvisibility = \"hidden\"\n")
            @test h.compile.visibility === :hidden
            @test CM.get_compile_flags(h) == ["-O2", "-fvisibility=hidden"]

            # Case-insensitive, like the other Symbol-valued keys.
            u = _cfg(dir, "c.toml", "visibility = \"HIDDEN\"\n")
            @test u.compile.visibility === :hidden
        end

        @testset "legacy raw flag still works, and never doubles" begin
            # argon2, box2d3 and zstd all carried `-fvisibility=hidden` in
            # [compile] flags before the key existed. That must keep working.
            legacy = _cfg(dir, "d.toml", "flags = [\"-O2\", \"-fvisibility=hidden\"]\n")
            @test legacy.compile.visibility === :default
            @test CM.get_compile_flags(legacy) == ["-O2", "-fvisibility=hidden"]

            # Key AND matching raw flag: one flag, not two.
            both = _cfg(dir, "e.toml",
                        "flags = [\"-O2\", \"-fvisibility=hidden\"]\nvisibility = \"hidden\"\n")
            @test count(==("-fvisibility=hidden"), CM.get_compile_flags(both)) == 1
        end

        @testset "the key reaches the compile fingerprint" begin
            # The load-bearing property: flipping the key must invalidate the
            # per-file IR cache. If this regresses, a package switched to hidden
            # reuses IR compiled visible and the wrapper keeps the old API.
            plain  = _cfg(dir, "f.toml", "flags = [\"-O2\"]\n")
            hidden = _cfg(dir, "g.toml", "flags = [\"-O2\"]\nvisibility = \"hidden\"\n")
            @test COMP.compute_compile_fingerprint(plain) !=
                  COMP.compute_compile_fingerprint(hidden)

            # ...and a package using the legacy spelling has the same fingerprint
            # as one using the key, since they compile identically.
            legacy = _cfg(dir, "h.toml", "flags = [\"-O2\", \"-fvisibility=hidden\"]\n")
            @test COMP.compute_compile_fingerprint(legacy) ==
                  COMP.compute_compile_fingerprint(hidden)
        end

        @testset "validation refuses rather than picking" begin
            p = joinpath(dir, "bad.toml")
            write(p, "[project]\nname = \"bad\"\nroot = \".\"\n\n[compile]\nvisibility = \"protected\"\n")
            err = try; CM.load_config(p); nothing catch e; e end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("visibility", msg)
            # Must point at the probe — choosing "hidden" blind is the whole hazard.
            @test occursin("visibility_probe", msg)

            # Key and raw flag asking for different things is undecidable, not a
            # precedence question.
            q = joinpath(dir, "conflict.toml")
            write(q, "[project]\nname = \"c2\"\nroot = \".\"\n\n[compile]\n" *
                     "flags = [\"-fvisibility=protected\"]\nvisibility = \"hidden\"\n")
            err2 = try; CM.load_config(q); nothing catch e; e end
            @test err2 isa ErrorException
            @test occursin("disagree", sprint(showerror, err2))
        end

        @testset "survives a round trip through save_config" begin
            h = _cfg(dir, "rt.toml", "flags = [\"-O2\"]\nvisibility = \"hidden\"\n")
            CM.save_config(h)
            @test CM.load_config(h.config_file).compile.visibility === :hidden
            @test occursin("visibility", read(h.config_file, String))

            # `:default` is the absent-key behaviour, so it is not written — every
            # generated TOML would otherwise carry a no-op line.
            d = _cfg(dir, "rt2.toml", "flags = [\"-O2\"]\n")
            CM.save_config(d)
            @test !occursin("visibility", read(d.config_file, String))
            @test CM.load_config(d.config_file).compile.visibility === :default
        end
    end

    @testset "survives discover(force=true)" begin
        mktempdir() do proj
            write(joinpath(proj, "lib.c"), "int f(int x){return x+1;}\n")
            toml = joinpath(proj, "replibuild.toml")
            write(toml, """
            [project]
            name = "visproj"
            root = "."

            [compile]
            flags = ["-O2", "-fPIC"]
            visibility = "hidden"
            """)

            kept = DISC._collect_preserved_sections(toml)
            @test kept !== nothing
            @test kept[("compile", "visibility")] == "hidden"

            # The real path: re-discovery regenerates the TOML from the tree, and
            # a tree cannot say whether the library annotates its exports.
            RepliBuild.discover(proj; force=true)
            @test CM.load_config(toml).compile.visibility === :hidden
            @test TOML.parsefile(toml)["compile"]["visibility"] == "hidden"
        end
    end
end
