#!/usr/bin/env julia
# test/test_cxx_personality.jl — the C++ personality name is one fact, stated in
# two languages, and they must agree.
#
# The dialect emits `llvm.invoke` from BOTH sides: the C++ lowering passes
# (`kCxxPersonality` in src/mlir/impl/JLCSPasses.cpp, mirrored in
# JLCSCAPIWrappers.cpp) and the Julia IR generator
# (`MLIRNative.CXX_PERSONALITY`, used by JLCSIRGenerator). Neither can see the
# other, so the name is written down twice — the "two derivations of one fact"
# hazard, with a quiet failure mode:
#
#   * Itanium C++ ABI unwinding is DWARF-based and calls `__gxx_personality_v0`.
#   * mingw-w64 on x86-64 unwinds with SEH and calls `__gxx_personality_seh0`,
#     and the C++ runtime there exports ONLY that one.
#
# So a disagreement costs exactly one undefined symbol at link time, on one
# platform, with a diagnostic that mentions neither exceptions nor Windows. It
# is invisible to any Linux run, which is why the check is textual rather than
# behavioural: the point is to catch the two definitions drifting apart.

using Test
using RepliBuild

const _SEH = "__gxx_personality_seh0"
const _V0 = "__gxx_personality_v0"

"""Both string literals from a `#if defined(_WIN32)` personality definition."""
function _cpp_personality_pair(path)
    src = read(path, String)
    win = match(Regex("defined\\(_WIN32\\)\\s*\\n\\s*static constexpr const char \\*kCxxPersonality = \"([^\"]+)\""), src)
    oth = match(Regex("#else\\s*\\n\\s*static constexpr const char \\*kCxxPersonality = \"([^\"]+)\""), src)
    return (win === nothing ? nothing : win.captures[1],
            oth === nothing ? nothing : oth.captures[1])
end

@testset "C++ personality name agrees across Julia and the dialect" begin
    mlir_dir = joinpath(dirname(@__DIR__), "src", "mlir", "impl")
    passes = joinpath(mlir_dir, "JLCSPasses.cpp")
    capi = joinpath(mlir_dir, "JLCSCAPIWrappers.cpp")
    @test isfile(passes)
    @test isfile(capi)

    @testset "the dialect picks SEH on Windows, Itanium elsewhere" begin
        for f in (passes, capi)
            (win, oth) = _cpp_personality_pair(f)
            @test win == _SEH
            @test oth == _V0
        end
    end

    @testset "Julia's constant matches what the dialect compiles to" begin
        expected = Sys.iswindows() ? _SEH : _V0
        @test RepliBuild.MLIRNative.CXX_PERSONALITY == expected
    end

    @testset "no emission site hardcodes the personality" begin
        # Every use must go through the constant, or the platform branch above
        # is decorative. A quoted literal is only legitimate in the definition
        # itself, so allow exactly the two lines of each #if/#else pair.
        offenders = String[]
        for f in (passes, capi)
            for (i, line) in enumerate(readlines(f))
                occursin("\"$_V0\"", line) || occursin("\"$_SEH\"", line) || continue
                occursin("kCxxPersonality =", line) && continue   # the definition
                startswith(strip(line), "//") && continue          # prose
                push!(offenders, string(basename(f), ":", i, "  ", strip(line)))
            end
        end
        if !isempty(offenders)
            @info "personality hardcoded outside its definition:\n  " *
                  join(offenders, "\n  ")
        end
        @test isempty(offenders)
    end

    @testset "the Julia generator emits the constant, not a literal" begin
        gen = read(joinpath(dirname(@__DIR__), "src", "IRGen", "JLCSIRGenerator.jl"), String)
        @test occursin("CXX_PERSONALITY", gen)
        # A literal here would silently win over the constant on Windows.
        @test !occursin("@$_V0", gen)
    end
end
