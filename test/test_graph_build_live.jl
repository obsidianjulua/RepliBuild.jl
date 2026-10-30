#!/usr/bin/env julia
# test/test_graph_build_live.jl — the graph executor (`Builder/GraphBuild.jl`)
# builds a two-library project described by hand as a TargetGraph (no cmake),
# and a function of one library that calls into the other is called from Julia.
#
# test/graph_build_test/:
#   alpha         C++ shared library, public header alpha/include/alpha.h,
#                 define ALPHA_FACTOR=3
#   beta          shared library linked against alpha, define BETA_OFFSET,
#                 include path beta/include + alpha/include
#   beta_helpers  OBJECT target folded into beta, define HELPER_BIAS=7
# Every source `#error`s on another target's define, so a unit compiled under
# the wrong group fails the build. beta_combine(5) = alpha_scale(5) [5*3] +
# beta_helper(5) [5*10+7] + BETA_OFFSET.
#
# test/graph_build_test/c/ is the same shape in C: cb linked against ca, both
# built in the C bucket (Julia's JLL clang, linked on Julia's libLLVM).
#
# The planning half (flags, folding, link inputs, refusals) is
# test_graph_build.jl, in runtests.

using Test
using TOML
using RepliBuild

isdefined(@__MODULE__, :TestSupport) || include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

if requires("Graph build (live)", :clangxx, :llvm_link)

const GBL = RepliBuild.GraphBuild
const GBL_BG = RepliBuild.BuildGraph
const GBL_DIR = replace(abspath(joinpath(@__DIR__, "graph_build_test")), '\\' => '/')

function gbl_graph(beta_offset::Int)
    G = GBL_BG
    cxx(defines, includes) = G.CompileGroup("CXX", ["-std=c++17", "-O2"], defines,
                                            [G.IncludeDir(GBL_DIR * "/" * i, false) for i in includes])
    alpha = G.GraphTarget("alpha::@1", "alpha", :shared,
        [G.GraphSource(GBL_DIR * "/alpha/src/alpha.cpp", 1, false),
         G.GraphSource(GBL_DIR * "/alpha/include/alpha.h", 0, false)],
        [cxx(["ALPHA_FACTOR=3"], ["alpha/include"])], String[], String[], String[])
    helpers = G.GraphTarget("beta_helpers::@1", "beta_helpers", :object,
        [G.GraphSource(GBL_DIR * "/beta/helpers/helpers.cpp", 1, false)],
        [cxx(["HELPER_BIAS=7"], String[])], String[], String[], String[])
    beta = G.GraphTarget("beta::@1", "beta", :shared,
        [G.GraphSource(GBL_DIR * "/beta/src/beta.cpp", 1, false),
         G.GraphSource(GBL_DIR * "/beta/include/beta.h", 0, false)],
        [cxx(["BETA_OFFSET=$beta_offset"], ["beta/include", "alpha/include"])],
        ["alpha::@1", "beta_helpers::@1"], String[], String[])
    return G.TargetGraph(GBL_DIR, GBL_DIR * "/build", [beta, helpers, alpha])
end

# Calls beta_combine in a child process, so the libraries are never loaded here
# (Windows would hold the files open across the rebuild below). On ELF beta
# finds alpha by its rpath alone; PE has none, so alpha is opened first, as a
# wrapper's __init__ opens a link_dirs DLL.
function gbl_call(dep_lib, lib, fn::Symbol, x)
    code = """
        import Libdl
        Sys.iswindows() && Libdl.dlopen($(repr(dep_lib)))
        h = Libdl.dlopen($(repr(lib)))
        println("PROBE call: ", ccall(Libdl.dlsym(h, $(repr(fn))), Cint, (Cint,), $x))
        println("PROBE_DONE")
        """
    r = run_isolated(`-e $code`; timeout = 300)
    v = probe_verdicts(r.output)
    return (r.ok && haskey(v, "PROBE_DONE")) ? tryparse(Int, get(v, "call", "")) : nothing
end

@testset "Graph build: two libraries and an object target, linked and called" begin
    out = replace(mktempdir(), '\\' => '/')
    results = GBL.build_graph(gbl_graph(100), out)
    r = Dict(x.name => x for x in results)
    @test sort(collect(keys(r))) == ["alpha", "beta", "beta_helpers"]

    alpha, beta, helpers = r["alpha"], r["beta"], r["beta_helpers"]
    @test alpha.status == :built && beta.status == :built
    @test alpha.bucket == :cpp && beta.bucket == :cpp
    @test isfile(alpha.library) && isfile(beta.library)
    @test alpha.dir == out * "/alpha" && startswith(beta.library, out * "/beta/lib/")
    @test helpers.status == :folded && helpers.folded_into == ["beta"]
    @test beta.link_dependencies == ["alpha::@1"]
    @test basename.(beta.sources) == ["beta.cpp", "helpers.cpp"]

    # Both of beta's units reached its linked module with their debug info.
    linked = read(only(beta.linked_ir), String)
    @test count("DICompileUnit(", linked) == 2
    @test occursin("beta.cpp", linked) && occursin("helpers.cpp", linked)

    manifest = TOML.parsefile(beta.dir * "/graph_target.toml")
    @test manifest["library"] == beta.library
    @test manifest["link_dependency_libraries"] == [alpha.library]
    @test manifest["sources"] == beta.sources
    @test manifest["defines"] == ["BETA_OFFSET=100", "HELPER_BIAS=7"]

    # beta_combine calls alpha_scale across the library boundary.
    @test gbl_call(alpha.library, beta.library, :beta_combine, 5) == 15 + 57 + 100

    # Unchanged, every unit comes from the IR cache.
    ir = vcat(alpha.ir_files, beta.ir_files)
    stamps = Dict(f => mtime(f) for f in ir)
    GBL.build_graph(gbl_graph(100), out)
    @test all(mtime(f) == t for (f, t) in stamps)

    # beta's define changes: beta.cpp recompiles, alpha's unit and the object
    # target's (other groups) do not, and the relinked beta sees the new value.
    results = GBL.build_graph(gbl_graph(200), out)
    beta2 = only(filter(x -> x.name == "beta", results))
    beta_ir = only(filter(f -> occursin("/beta.cpp-", f), beta2.ir_files))
    @test mtime(beta_ir) != stamps[beta_ir]
    @test all(mtime(f) == stamps[f] for f in ir if f != beta_ir)
    @test gbl_call(alpha.library, beta2.library, :beta_combine, 5) == 15 + 57 + 200
end

@testset "Graph build: two C libraries in the C bucket" begin
    G = GBL_BG
    c = GBL_DIR * "/c"
    cgroup(defines) = G.CompileGroup("C", ["-std=gnu11", "-O2"], defines, [G.IncludeDir(c, false)])
    ca = G.GraphTarget("ca", "ca", :shared, [G.GraphSource(c * "/ca.c", 1, false)],
                       [cgroup(String[])], String[], String[], String[])
    cb = G.GraphTarget("cb", "cb", :shared, [G.GraphSource(c * "/cb.c", 1, false)],
                       [cgroup(["CB_K=1000"])], ["ca"], String[], String[])
    out = replace(mktempdir(), '\\' => '/')
    r = Dict(x.name => x for x in GBL.build_graph(G.TargetGraph(c, c * "/build", [cb, ca]), out))
    @test r["ca"].bucket == :c && r["cb"].bucket == :c
    @test r["cb"].link_dependencies == ["ca"]
    # Julia's own clang built it: its producer carries the `jl` suffix.
    linked = read(only(r["cb"].linked_ir), String)
    @test occursin("producer: \"clang version $(Base.libllvm_version)jl", linked)
    # Both calls reach ca's one counter: (1 + 1) + (1 + 2) + 1000.
    @test gbl_call(r["ca"].library, r["cb"].library, :cb_twice, 1) == 1005
end

end # requires
