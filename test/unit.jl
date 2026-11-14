#!/usr/bin/env julia
# RepliBuild.jl — unit suite: every toolchain-free regression test, in one
# process. The files are listed in test/suites.jl under `:unit`; this is
# `julia --project=. test/run.jl unit`, so `--changed`, `--area=` and
# `--files=` work the same way from run.jl.
#
#     julia --project=. test/unit.jl                 # all of it
#     julia --project=. test/run.jl unit --changed   # the files a diff touches
#
# Needs no C/C++ toolchain. What `Pkg.test()` runs is test/runtests.jl, the
# registration basics; the full set of what compiles and JITs is devtests.jl.

include(joinpath(@__DIR__, "run.jl"))
exit(main(vcat(["unit"], ARGS)))
