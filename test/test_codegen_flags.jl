# test_codegen_flags.jl — IR is turned into machine code at the link level.
#
# create_library handed clang `-shared -fPIC` and the optimized IR, with no -O:
# the -O0 backend (FastISel + the fast register allocator) generated every
# shipped library, and on LLVM 23 crashed outright ("Cannot emit physreg copy
# instruction", llama.cpp on Windows). Codegen now runs at [link]
# optimization_level with the IR pipeline disabled, since that IR is already
# optimized. No toolchain.

using Test
using RepliBuild

const _CG_CM = RepliBuild.ConfigurationManager
const _CG_C  = RepliBuild.Compiler

function _cg_config(level::String)
    dir = mktempdir()
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "cgprobe"
    root = "."
    [link]
    optimization_level = "$level"
    """)
    return _CG_CM.load_config(toml)
end

@testset "codegen opt flags" begin
    for (level, flag) in (("0", "-O0"), ("2", "-O2"), ("3", "-O3"), ("s", "-Os"), ("z", "-Oz"))
        @test _CG_C._codegen_opt_flags(_cg_config(level)) == [flag, "-Xclang", "-disable-llvm-passes"]
    end

    # Both IR -> machine code sites use it: the merged-module link and the per-TU
    # fallback. Read from the source, the way the config-surface guard does.
    src = read(joinpath(pkgdir(RepliBuild), "src", "Builder", "Compiler.jl"), String)
    @test occursin("_codegen_opt_flags(config)...,", src)                       # create_library
    @test occursin(r"vcat\(\[\"-c\", \"-fPIC\"\], _codegen_opt_flags\(config\)", src)  # per-TU fallback
end
