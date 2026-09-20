# test_wrap_surface_guard.jl — `Compiler.verify_wrap_surface`
#
# The guard asserts that every function metadata calls wrappable is reachable through
# the DYNAMIC symbol table the generated wrapper will dlsym against. Two tables feed
# the pipeline — `nm -g` for metadata, `nm -D` for the wrapper — and nothing checked
# that they agree.
#
# The failure this exists for is `-fvisibility=hidden` × `[link] promote_statics`:
# hidden visibility is how a library marks internals, promotion renames hidden
# definitions to `__rb_*` so they drop out of the wrap surface, and that is correct
# until the flag makes EVERY unannotated function hidden — including public API whose
# export macro is a bare `extern` (lua's `LUA_API`). The API is then renamed away and
# the wrapper generates nothing, from a build that succeeded.
#
# Fixture is written at runtime into a tempdir: library-free, three functions, one per
# annotation shape. Nothing tracked, so a fresh clone cannot hit the gitignored-toml
# trap that strands the slice_test suites.
#
# Needs the JLL clang (always present — it is a dependency, not a system toolchain)
# and `nm`, which the build path already requires.

using Test
using RepliBuild
using JSON

const C = RepliBuild.Compiler
const CM = RepliBuild.ConfigurationManager

# One function per shape the flag distinguishes.
const FIXTURE_C = """
/* Annotated export — the shape 17 of 26 Hub upstreams use. Survives -fvisibility=hidden. */
#define PUB __attribute__((visibility("default")))
PUB int annotated_public(int x) { return x + 1; }

/* Bare `extern` export — lua's LUA_API. PUBLIC API that goes hidden under the flag. */
extern int bare_extern_public(int x);
int bare_extern_public(int x) { return x + 2; }

/* A genuine internal. Correct to hide and promote. */
static int helper(int x) { return x * 3; }
int uses_helper(int x) { return helper(x); }
"""

"""
Build the fixture to a `.so`, optionally under `-fvisibility=hidden`, optionally
running the real static-promotion pass. Returns the `.so` path, or `nothing` if the
toolchain could not produce one (skip rather than fail — this file must not be the
thing that breaks on a host without a linker).
"""
function build_fixture(dir::String; hidden::Bool, promote::Bool)
    src = joinpath(dir, "lib.c")
    write(src, FIXTURE_C)
    tag = (hidden ? "hidden" : "default") * (promote ? "_promoted" : "")
    ll  = joinpath(dir, "lib_$tag.ll")

    flags = ["-S", "-emit-llvm", "-g", "-O2", "-fPIC"]
    hidden && push!(flags, "-fvisibility=hidden")
    (_, rc) = C._clang_for_c_bucket("clang", vcat(flags, ["-o", ll, src]))
    rc == 0 && isfile(ll) || return nothing

    if promote
        out = joinpath(dir, "lib_$(tag)_p.ll")
        C._promote_statics_libllvm(ll, out, "fix") === nothing && return nothing
        ll = out
    end

    so = joinpath(dir, "lib$tag.so")
    (_, rc2) = C._clang_for_c_bucket("clang", ["-shared", "-fPIC", "-o", so, ll])
    rc2 == 0 && isfile(so) ? so : nothing
end

"""Write a `compilation_metadata.json` naming `names` as wrappable functions."""
function write_metadata(dir::String, names::Vector{String})
    path = joinpath(dir, "compilation_metadata.json")
    open(path, "w") do io
        JSON.print(io, Dict(
            "functions" => [Dict("name" => n, "mangled" => n, "demangled" => n,
                                 "exported" => true) for n in names],
        ))
    end
    return path
end

"""Minimal shared-library config; the guard reads only `binary.type`."""
function fixture_config(dir::String; binary_type::Symbol=:shared)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "wsfix"
    root = "."

    [binary]
    type = "$(binary_type)"
    """)
    return CM.load_config(toml)
end

dynsym(so) = Set(String(p[3]) for p in
                 (split(strip(l)) for l in split(read(`nm -D --defined-only $so`, String), '\n'))
                 if length(p) >= 3)

@testset "verify_wrap_surface" begin
    mktempdir() do dir
        healthy = build_fixture(dir; hidden=false, promote=true)

        if healthy === nothing
            @test_skip "JLL clang could not produce a .so on this host"
        else
            cfg = fixture_config(dir)

            @testset "the fixture reproduces the class" begin
                # Without this the rest of the file could pass against a fixture that
                # stopped exercising anything — the vacuous-green shape.
                syms = dynsym(healthy)
                @test "annotated_public" in syms
                @test "bare_extern_public" in syms
                @test "uses_helper" in syms

                collapsed_so = build_fixture(dir; hidden=true, promote=true)
                @test collapsed_so !== nothing
                csyms = dynsym(collapsed_so)
                # The annotation is what survives the flag; a bare `extern` does not.
                @test "annotated_public" in csyms
                @test !("bare_extern_public" in csyms)
                @test "__rb_fix_bare_extern_public" in csyms
                @test "__rb_fix_uses_helper" in csyms
            end

            @testset "R2 — healthy library passes" begin
                md = write_metadata(dir, ["annotated_public", "bare_extern_public",
                                          "uses_helper"])
                @test C.verify_wrap_surface(cfg, healthy, md) === nothing
            end

            @testset "R2 — unreachable public API is refused" begin
                collapsed = build_fixture(dir; hidden=true, promote=true)
                md = write_metadata(dir, ["annotated_public", "bare_extern_public",
                                          "uses_helper"])
                err = try
                    C.verify_wrap_surface(cfg, collapsed, md); nothing
                catch e; e end
                @test err isa ErrorException
                msg = sprint(showerror, err)
                @test occursin("NOT reachable", msg)
                # Names the offenders, and does NOT name the one that is fine.
                @test occursin("bare_extern_public", msg)
                @test occursin("uses_helper", msg)
                @test !occursin("→ annotated_public", msg)
                # Points at both halves of the cause.
                @test occursin("visibility", msg)
                @test occursin("promote_statics", msg)
            end

            @testset "R1 — empty surface is refused" begin
                collapsed = build_fixture(dir; hidden=true, promote=true)
                md = write_metadata(dir, String[])
                err = try
                    C.verify_wrap_surface(cfg, collapsed, md); nothing
                catch e; e end
                @test err isa ErrorException
                @test occursin("EMPTY", sprint(showerror, err))
            end

            @testset "must not fire" begin
                md = write_metadata(dir, ["annotated_public"])
                # An executable is not wrapped; the guard has no opinion.
                @test C.verify_wrap_surface(fixture_config(dir; binary_type=:executable),
                                            healthy, md) === nothing
                # Unaskable questions assert nothing rather than failing.
                @test C.verify_wrap_surface(cfg, healthy, nothing) === nothing
                @test C.verify_wrap_surface(cfg, healthy,
                                            joinpath(dir, "absent.json")) === nothing
                @test C.verify_wrap_surface(cfg, joinpath(dir, "absent.so"), md) === nothing
                # Metadata with no functions and no promoted symbols is not a collapse.
                plain = build_fixture(dir; hidden=false, promote=false)
                if plain !== nothing
                    @test C.verify_wrap_surface(cfg, plain,
                                                write_metadata(dir, String[])) === nothing
                end
            end
        end
    end
end
