#!/usr/bin/env julia
# test/test_c_bucket_sysroot.jl — the C bucket's JLL clang must be given a libc
# on Windows.
#
# The C bucket compiles .c with `Clang_unified_jll.clang()` on purpose: that
# tool is version-locked to Julia's resident libLLVM, which is the invariant the
# bucket exists to hold (see _clang_for_c_bucket). On Linux that costs nothing,
# because the JLL compiler falls back to the system's /usr/include.
#
# On Windows there is no such fallback. The JLL is a bare compiler artifact —
# its InstalledDir is a Julia artifact directory holding tools and nothing else
# — so it ships no libc headers and no mingw CRT. Its default target is already
# `x86_64-w64-windows-gnu`, so the triple is right and the diagnostic is not
# about triples at all; every C file simply dies on its first system include:
#
#     mathkit.c:1:10: fatal error: 'math.h' file not found
#
# `--sysroot=<MSYS2 CLANG64 prefix>` supplies the missing half and moves nothing
# else: the same JLL clang emits the IR, Julia's libLLVM still consumes it.
#
# The functional half of this test only runs on Windows with the JLL present.
# The invariants below it are checked everywhere, because they are the parts a
# Linux-only edit can silently break.

using Test
using RepliBuild

const _CB = RepliBuild.Compiler

@testset "C bucket sysroot" begin
    sysroot = _CB._c_bucket_sysroot()

    if !Sys.iswindows()
        # No sysroot is injected off Windows — the system compiler finds its own
        # headers, and forcing a prefix there would be a regression, not a fix.
        @test sysroot == ""
    else
        # Memoized: the search is identical within a session, and repeating it
        # would repeat the not-found warning once per compiled file.
        @test _CB._c_bucket_sysroot() === sysroot

        if isempty(sysroot)
            @warn "no mingw sysroot on this machine — skipping the C bucket compile"
        else
            # Whatever was chosen must be a real sysroot, not merely a directory
            # that exists. An MSYS2 install carries empty ucrt64/ and mingw64/
            # trees for environments never installed; accepting one of those
            # trades "no sysroot" for "wrong sysroot".
            @test isfile(joinpath(sysroot, "include", "math.h"))

            mktempdir() do d
                cfile = joinpath(d, "sysroot_probe.c")
                write(cfile, """
                      #include <math.h>
                      #include <stdio.h>
                      double rb_sysroot_probe(double a, double b) { return sqrt(a*a + b*b); }
                      """)
                ll = joinpath(d, "sysroot_probe.ll")

                # Exactly the call the build makes — not a hand-rolled clang
                # invocation, or the test would pass while the product failed.
                out, code = _CB._clang_for_c_bucket(
                    "clang", ["-S", "-emit-llvm", "-g", "-o", ll, cfile])

                code == 0 || @info "C bucket compile output:\n$out"
                @test code == 0
                @test isfile(ll)

                ir = read(ll, String)
                @test occursin("rb_sysroot_probe", ir)
                # The sysroot must not have dragged the target off the mingw
                # triple; a wrong triple here would poison every later stage.
                @test occursin("x86_64-w64-windows-gnu", ir)

                # A caller that names its own sysroot keeps it. This is the
                # override path, so it has to lose to the caller, not win.
                _, code2 = _CB._clang_for_c_bucket(
                    "clang", ["--sysroot=$(joinpath(d, "no-such-sysroot"))",
                              "-S", "-emit-llvm", "-o", joinpath(d, "x.ll"), cfile])
                @test code2 != 0
            end

            # The shim-header guard vouches for the real build's include
            # resolution, which it can only do if it compiles with the same
            # sysroot. Drift here makes the guard report on headers no build
            # ever sees.
            @test occursin("--sysroot=$sysroot", string(_CB._probe_compiler(:c)))
        end

        # C++ goes through the system clang++, which has its own headers; it
        # must not be handed the C bucket's sysroot.
        @test !occursin("--sysroot", string(_CB._probe_compiler(:cpp)))
    end
end
