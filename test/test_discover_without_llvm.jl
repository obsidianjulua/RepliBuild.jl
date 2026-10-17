# test_discover_without_llvm.jl — discover works on a machine with no system LLVM.
#
# A C project needs nothing but Julia: the C bucket compiles with the Clang JLL
# and links in-process (install.md). discover asked LLVMEnvironment for a
# toolchain twice, for its optional dependency walk and to warm the toolchain for
# a later build, and with no LLVM at all both raised "LLVM Toolchain not found"
# instead of degrading. The first CI run on GitHub found it: the Ubuntu runner has
# no LLVM that RepliBuild accepts, and test_registry's c_test integration runs
# discover on a fresh clone.
#
# The child process replaces get_llvm_root with one that finds nothing, so every
# caller above it runs for real, on any host, including one with LLVM installed.
# REPLIBUILD_HOME keeps discover's auto-registration out of the real registry.
# No toolchain.

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

@testset "discover without a system LLVM" begin
    mktempdir() do dir
        mkpath(joinpath(dir, "src"))
        write(joinpath(dir, "src", "tiny.c"), "int tiny_add(int a, int b) { return a + b; }\n")
        code = """
        ENV["REPLIBUILD_HOME"] = mktempdir()
        using RepliBuild
        const LE = RepliBuild.LLVMEnvironment
        @eval LE get_llvm_root(source::Symbol=:auto; config=nothing) =
            error("LLVM Toolchain not found. (simulated: no system LLVM)")
        println("TOOLCHAIN ", try LE.get_tool("clang++"); "visible" catch; "hidden" end)
        toml = RepliBuild.discover(ARGS[1]; force = true)
        cfg = RepliBuild.ConfigurationManager.load_config(toml)
        println("DISCOVERED ", isfile(toml), " ", join(basename.(cfg.compile.source_files), ","))
        """
        r = run_isolated(`-e $code $dir`; timeout = 300)
        @test occursin("TOOLCHAIN hidden", r.output)   # the case under test was set up
        @test r.ok
        m = match(r"DISCOVERED (\w+) (\S*)", r.output)
        @test m !== nothing
        m === nothing || @test (m.captures[1], m.captures[2]) == ("true", "tiny.c")
        @test occursin("skipping it", r.output)        # it said why there is no dependency graph
    end
end
