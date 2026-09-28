#!/usr/bin/env julia
# test/test_c_bucket_pipe.jl — the C-bucket clang call drains its output while
# clang runs.
#
# `_clang_for_c_bucket` read its `Pipe`s after `run` returned. Once clang wrote
# more than the OS pipe buffer (~64 KiB of diagnostics: a library with a few
# thousand warnings), clang blocked on the write, Julia blocked on clang's exit,
# and the build hung with no error and no output (audit F7, 2026-09-26). It
# drains into IOBuffers while the process runs now.
#
# The failure mode is a hang, so the call runs in a child process with a SIGKILL
# timeout: a regression is a timed-out child, not a wedged suite. Negative-checked
# by re-defining the old Pipe-then-read body in the child: it is killed at the
# timeout with 0 bytes read. Needs only the JLL clang RepliBuild depends on.

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

@testset "C-bucket clang drains more than a pipe buffer" begin
    mktempdir() do dir
        src = joinpath(dir, "noisy.c")
        open(src, "w") do io
            for i in 1:4000
                println(io, "#warning \"deliberately long warning number $i to overflow a 64 KiB pipe buffer quickly\"")
            end
            println(io, "int f(void){return 1;}")
        end
        code = """
        using RepliBuild
        out, rc = RepliBuild.Compiler._clang_for_c_bucket("clang", ["-fsyntax-only", ARGS[1]])
        println("RC=", rc, " BYTES=", sizeof(out), " WARNINGS=", count("warning: \\"deliberately", out))
        """
        r = run_isolated(`-e $code $src`; timeout = 300)
        @test !r.timedout            # the regression: clang blocked on a full pipe
        @test r.ok
        m = match(r"RC=(-?\d+) BYTES=(\d+) WARNINGS=(\d+)", r.output)
        @test m !== nothing
        if m !== nothing
            rc, bytes, warns = parse.(Int, m.captures)
            @test rc == 0                 # -fsyntax-only, warnings only
            @test bytes > 64 * 1024       # the case under test did exceed a pipe buffer
            @test warns == 4000           # and draining it lost nothing
        end
    end
end
