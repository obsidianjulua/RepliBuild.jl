# =============================================================================
# Supplementary header enums come from the project's own headers
#
# `ClangJLBridge.extract_header_types` adds enums the compiler eliminated (never
# used, so not in DWARF) by walking the headers with Clang.jl. The walk used to
# take every `EnumDecl` in the translation unit, so a header that included
# <system_error> or <charconv> brought libstdc++'s `errc` and `_Integer_kind`
# into the wrapper, and which ones depended on the libclang Clang.jl ships for
# the running Julia (LLVM 18 on 1.12, 20 on 1.13): pugixml's wrapper differed by
# 101 lines between the two. Only declarations under the directory of a named
# header or under an include dir are the library's.
#
# No system headers are involved: the "foreign" header sits in a third
# directory and is included by absolute path, so the check does not depend on
# libstdc++ being found by the subprocess.
# =============================================================================

using Test
using RepliBuild

const _CJB = RepliBuild.ClangJLBridge

@testset "header enums: only declarations under the project's headers or include dirs" begin
    mktempdir() do dir
        inc, inc2, foreign = joinpath(dir, "inc"), joinpath(dir, "inc2"), joinpath(dir, "elsewhere")
        mkpath(inc); mkpath(inc2); mkpath(foreign)
        write(joinpath(foreign, "foreign.h"), "enum Foreign { F_A = 1, F_B = 2 };\n")
        write(joinpath(inc2, "shared.h"), "enum Shared { S_X = 7 };\n")
        write(joinpath(inc, "api.h"), """
            #include "$(escape_string(joinpath(foreign, "foreign.h")))"
            #include <shared.h>
            enum Color { RED = 0, GREEN = 1, BLUE = 2 };
            namespace ns { enum class Mode { Fast = 3 }; }
            """)
        enums = _CJB.extract_header_types([joinpath(inc, "api.h")], [inc2])["enums"]
        # The walk ran: a vacuous empty result would pass the negative assertion alone.
        @test haskey(enums, "Color")
        @test sort(get(enums, "Color", Tuple{String,Int}[])) == [("BLUE", 2), ("GREEN", 1), ("RED", 0)]
        @test haskey(enums, "Shared")        # reached through an include dir
        @test haskey(enums, "Mode")          # namespaces are walked
        @test !haskey(enums, "Foreign")      # declared outside every scope dir
    end
end
