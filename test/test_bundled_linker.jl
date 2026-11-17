# test_bundled_linker.jl — which linker the C bucket's final link runs.
#
# The C bucket's library was linked by whatever `ld` the clang driver found on
# PATH. It is now `ld.lld` from LLD_unified_jll, at the LLVM of the bundled
# clang, selected with `--ld-path` (Compiler._bundled_linker). This file holds
# the policy: bundled only for C, on Linux, when the JLL clang compiles and
# `Tools` took the JLL's copy or an explicit override; anything else is the
# driver's own choice, and which of the two it was is part of the toolchain
# identity a project hash carries. test_bundled_linker_live.jl links with it.
#
# No toolchain: nothing here starts a compiler.

using Test
using RepliBuild

const _BL_C = RepliBuild.Compiler
const _BL_T = RepliBuild.Tools

# The JLL clang being there or not is injected: the policy, not this host.
_bl_have() = `clang`
_bl_none() = nothing

@testset "the bundled linker is the C bucket's, on Linux, beside the JLL clang" begin
    withenv("REPLIBUILD_TOOLS" => "jll", "REPLIBUILD_TOOL_LD_LLD" => nothing) do
        lld = _BL_C._bundled_linker(:c; jll_clang = _bl_have)
        if Sys.islinux()
            @test lld == _BL_T.tool_path("ld.lld")
            @test basename(lld) == "ld.lld"       # lld picks its flavour from this name
            @test isfile(lld)
            @test _BL_T.tool_source("ld.lld") === :jll
        else
            @test lld === nothing
        end
        # C++ links with the system toolchain it compiles with.
        @test _BL_C._bundled_linker(:cpp; jll_clang = _bl_have) === nothing
        # No JLL clang: the system clang would start LLD without its libLLVM.
        @test _BL_C._bundled_linker(:c; jll_clang = _bl_none) === nothing
    end
end

@testset "REPLIBUILD_TOOLS=system is the system's default linker" begin
    withenv("REPLIBUILD_TOOLS" => "system", "REPLIBUILD_TOOL_LD_LLD" => nothing) do
        # Not an ld.lld that happens to be on PATH: no flag at all.
        @test _BL_C._bundled_linker(:c; jll_clang = _bl_have) === nothing
    end
    withenv("REPLIBUILD_TOOLS" => "system", "REPLIBUILD_TOOL_LD_LLD" => "/opt/x/ld.lld") do
        # An explicit override is asked for by name and wins either way.
        @test _BL_C._bundled_linker(:c; jll_clang = _bl_have) == (Sys.islinux() ? "/opt/x/ld.lld" : nothing)
    end
end

@testset "the linker is part of what was built" begin
    with_lld = withenv(() -> _BL_C._toolchain_identity(:c), "REPLIBUILD_TOOLS" => "jll")
    without  = withenv(() -> _BL_C._toolchain_identity(:c), "REPLIBUILD_TOOLS" => "system",
                       "REPLIBUILD_TOOL_LD_LLD" => nothing)
    bundled = withenv(() -> _BL_C._bundled_linker(:c), "REPLIBUILD_TOOLS" => "jll") !== nothing
    @test endswith(with_lld, " | ld.lld") == bundled
    @test !occursin("ld.lld", without)
    @test (with_lld != without) == bundled
    # A C++ project's identity never names it.
    cxx = try
        withenv(() -> _BL_C._toolchain_identity(:cpp), "REPLIBUILD_TOOLS" => "jll")
    catch
        ""        # no system clang++ on this host: nothing to name
    end
    @test !occursin("ld.lld", cxx)
end

@testset "the C fallback's llvm-link and opt are at Julia's LLVM, from the package" begin
    # `[link] fallback = true` runs external llvm-link and opt, and for C they
    # must be at the major of Julia's libLLVM. A machine whose only system
    # LLVM is newer had none; LLVM_unified_jll's are the last place looked.
    dir = _BL_T.matched_llvm_bin()
    @test dir === nothing || isdir(dir)
    if dir !== nothing
        exe = Sys.iswindows() ? ".exe" : ""
        checked = 0
        for tool in ("llvm-link", "opt")
            path = joinpath(dir, tool * exe)
            @test isfile(path)
            @test _BL_T.in_bundled_dir(path)
            out, code = RepliBuild.BuildBridge.execute(path, ["--version"])
            @test code == 0
            m = match(r"LLVM version (\d+)\.", out)
            @test m !== nothing && parse(Int, m.captures[1]) == Base.libllvm_version.major
            checked += 1
        end
        @test checked == 2
        # So the hatch has a toolchain wherever the JLL has a build.
        @test RepliBuild.LLVMEnvironment.c_toolchain_bin_dir() !== nothing
        # A program from anywhere else is left alone.
        @test !_BL_T.in_bundled_dir("/usr/bin/opt")
        other = `/usr/bin/opt --version`
        @test _BL_T.with_dir_env(other) === other
    end
end
