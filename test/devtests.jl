#!/usr/bin/env julia
# RepliBuild.jl — dev suite: everything that compiles, links or JITs
# (test/run.jl). Requires LLVM 21+, Clang, CMake and libJLCS (src/mlir/build.sh)
# on the system; files needing a tool that is absent record a skipped testset
# (TestSupport.requires).
#
#     julia --project=. test/devtests.jl                     # every job (~10 min on 12 threads)
#     julia --project=. test/devtests.jl --changed           # what the diff against origin/main touches
#     julia --project=. test/devtests.jl --area=c            # one area: build, dwarf, c, cpp, dialect, graph
#     julia --project=. test/devtests.jl --files=test_c_sret_abi.jl
#     julia --project=. test/devtests.jl --changed --processes --tail   # a child per job, logs printed
#
# The jobs, their areas and what they lock or run after are in test/suites.jl;
# a fixture job (`fixture:<name>`) is test/support/fixture.jl. One process by
# default (test/run.jl says why); `--processes` gives a child per job with a log
# in test/.logs/. REPLIBUILD_SKIP_TESTS names fixtures to skip.

include(joinpath(@__DIR__, "run.jl"))
exit(main(vcat(["dev"], ARGS)))
