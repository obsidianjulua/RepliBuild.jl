# RepliBuild.jl — C-bucket in-process libLLVM pipeline
#
# Traces the C link/optimize path through Julia's resident libLLVM (the
# default for C: `[link] fallback = false`) and the external llvm-link/opt
# escape hatch (`fallback = true`), asserting at each stage that DWARF survives
# — the property the in-process path must not silently break. Requires a C
# toolchain (run under devtests.jl, not the lightweight CI runtests.jl).

using Test
using RepliBuild
using TOML

const _CINP_TEST_DIR = @__DIR__

"""Write a c_test-derived toml into `root` with the given link overrides."""
function _c_inprocess_toml(root::String; opt_level::String, fallback::Union{Bool,Nothing})
    base = TOML.parsefile(joinpath(_CINP_TEST_DIR, "c_test", "replibuild.toml"))
    base["link"]["optimization_level"] = opt_level
    fallback === nothing ? delete!(base["link"], "fallback") : (base["link"]["fallback"] = fallback)
    name = "c_inproc_" * basename(root)
    base["project"]["root"] = root
    base["project"]["name"] = name
    base["cache"]["directory"] = joinpath(root, ".replibuild_cache")
    mkpath(root)
    toml_path = joinpath(root, "replibuild.toml")
    open(toml_path, "w") do io; TOML.print(io, base); end
    return toml_path, name
end

"""True if `so` carries DWARF subprogram DIEs."""
function _has_dwarf(so::String)
    out = read(`llvm-dwarfdump --debug-info $so`, String)
    return occursin("DW_TAG_subprogram", out)
end

@testset "C in-process libLLVM pipeline" begin
    Compiler = RepliBuild.Compiler

    @testset "config: fallback defaults false, round-trips" begin
        toml, _ = _c_inprocess_toml(mktempdir(); opt_level="0", fallback=nothing)
        c = RepliBuild.ConfigurationManager.load_config(toml)
        @test c.link.fallback == false           # default

        toml2, _ = _c_inprocess_toml(mktempdir(); opt_level="0", fallback=true)
        c2 = RepliBuild.ConfigurationManager.load_config(toml2)
        @test c2.link.fallback == true           # explicit parse
    end

    @testset "in-process link (O0) preserves DWARF" begin
        root = mktempdir()
        toml, name = _c_inprocess_toml(root; opt_level="0", fallback=false)
        RepliBuild.build(toml)
        so = joinpath(root, "julia", "lib$(name).so")
        @test isfile(so)
        @test _has_dwarf(so)
    end

    @testset "in-process opt (O2) preserves DWARF end-to-end" begin
        root = mktempdir()
        toml, name = _c_inprocess_toml(root; opt_level="2", fallback=false)
        RepliBuild.build(toml)
        RepliBuild.wrap(toml)
        so = joinpath(root, "julia", "lib$(name).so")
        @test isfile(so)
        @test _has_dwarf(so)
    end

    @testset "_optimize_ir_libllvm: in-process opt keeps debug metadata" begin
        # Build once at O0 to get a linked .ll, then optimize it in-process.
        root = mktempdir()
        toml, name = _c_inprocess_toml(root; opt_level="0", fallback=false)
        RepliBuild.build(toml)
        linked = filter(f -> endswith(f, "_linked.ll"),
                        readdir(joinpath(root, "build"), join=true))
        @test !isempty(linked)
        out_ll = joinpath(root, "opt2.ll")
        @test Compiler._optimize_ir_libllvm(first(linked), out_ll, "2")
        txt = read(out_ll, String)
        @test occursin("DICompileUnit", txt) || occursin("!llvm.dbg", txt)
    end

    @testset "external escape hatch (fallback=true) still builds with DWARF" begin
        root = mktempdir()
        toml, name = _c_inprocess_toml(root; opt_level="2", fallback=true)
        RepliBuild.build(toml)
        so = joinpath(root, "julia", "lib$(name).so")
        @test isfile(so)
        @test _has_dwarf(so)
    end
end
