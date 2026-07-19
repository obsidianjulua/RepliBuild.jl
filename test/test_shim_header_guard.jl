# Macro-shim header-collision guard — library-free fixture.
#
# Reproduces, with NO real library in play, the failure BLAKE3 hit in the Hub:
# the generated macro-shim TU lives outside the vendored source tree, so a bare
# `#include "foo.h"` can silently resolve to a system-installed copy of foo.h at a
# DIFFERENT version — baking wrong [wrap.macros] values with no other symptom.
#
# The whole point of the hub-wrap guard is that a structural fix must reproduce on
# a minimal hand-written fixture with the candidate library removed. This does that:
# two temp trees and a header basename that exists in both. It exercises the exact
# resolution path (clang -H) the real shim compile uses.

using Test
using RepliBuild

const _C = RepliBuild.Compiler

@testset "Macro-shim header collision guard" begin
    proj   = mktempdir()          # stands in for the project / dependency tree
    system = mktempdir()          # stands in for /usr/include (OUTSIDE the tree)

    mkpath(joinpath(proj, "inc"))
    write(joinpath(proj, "inc", "collide.h"), "#define LIB_VERSION \"VENDORED\"\n")
    write(joinpath(system, "collide.h"),      "#define LIB_VERSION \"SYSTEM\"\n")

    # Stand-in for the generated shim: its only direct include is the header, and
    # it sits under .replibuild_cache (outside the header's own directory), exactly
    # like the real replibuild_shims.c.
    shim = joinpath(proj, ".replibuild_cache", "replibuild_shims.c")
    mkpath(dirname(shim))
    write(shim, "#include \"collide.h\"\nint shim_LIB_VERSION(void){return 0;}\n")

    probe   = _C._probe_compiler(:c)
    allowed = [proj]              # only the project tree is in-bounds

    @testset "reproduction — foreign header wins silently" begin
        # Only the out-of-tree dir is on the include path: the shim resolves the
        # foreign copy. This IS the BLAKE3 misresolution, minus BLAKE3.
        offenders = _C._shim_headers_out_of_tree(shim, probe, ["-I$system"], allowed)
        @test !isempty(offenders)
        @test any(o -> startswith(o, realpath(system)), offenders)
    end

    @testset "clean — vendored header is found in-tree" begin
        offenders = _C._shim_headers_out_of_tree(
            shim, probe, ["-I$(joinpath(proj, "inc"))"], allowed)
        @test isempty(offenders)
    end

    @testset "clean — vendored -I precedes system -I (the lua/sqlite case)" begin
        # clang searches -I dirs in order; vendored first ⇒ vendored wins. This is
        # why packages whose header sits at the auto-included clone root never trip.
        offenders = _C._shim_headers_out_of_tree(
            shim, probe, ["-I$(joinpath(proj, "inc"))", "-I$system"], allowed)
        @test isempty(offenders)
    end

    @testset "a missing header is NOT a collision" begin
        # No -I ⇒ collide.h is unresolvable ⇒ clang errors, emits no depth-1 include.
        # The guard must stay silent (that is a compile error to surface elsewhere,
        # not a header-collision false positive).
        offenders = _C._shim_headers_out_of_tree(shim, probe, String[], allowed)
        @test isempty(offenders)
    end
end
