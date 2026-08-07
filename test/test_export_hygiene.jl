# =============================================================================
# Export hygiene — a generated module must not shadow Base in its CONSUMER
# =============================================================================
#
# A wrapper's namespace belongs to the library, and its export list is harvested
# from every symbol that reached the debug info. Several of those collide with
# Base/Core exports: llamacpp exports `all`, `error`, `stat`, `symlink`; sqlite
# exports `Expr`, `Module`, `stat`; cjson exports `error`.
#
# `_assert_base_calls_qualified` protects the wrapper's OWN calls. This is the
# other half: `using` such a module used to shadow the caller's `error`, so
# every failure path in the CALLER raised `UndefVarError: error not defined`
# instead of its message — invisible until something went wrong, since the
# happy path never touches the shadowed binding.
#
# No toolchain: the emitter is a pure function of a name list, and the live
# check builds a module from the emitted text with include_string.

using Test

@testset "Export hygiene" begin
    W = RepliBuild.Wrapper

    @testset "shadow detection" begin
        # Base exports, Core exports, and names that collide with neither.
        @test W._base_shadowing(["error", "stat", "all", "symlink"]) ==
              ["all", "error", "stat", "symlink"]
        @test "Expr"   in W._base_shadowing(["Expr"])     # Core, re-exported
        @test "Module" in W._base_shadowing(["Module"])
        @test isempty(W._base_shadowing(["llama_decode", "lua_pushnil", "z_deflate"]))
        # Sorted + deduplicated, so the emitted banner is stable across runs.
        @test W._base_shadowing(["stat", "error", "stat"]) == ["error", "stat"]
    end

    @testset "export statement" begin
        # No collision → a bare export line, byte-identical to the old emission.
        @test W._export_statement(["foo", "bar"]) == "export foo, bar\n\n"
        @test W._export_statement(String[]) == ""

        s = W._export_statement(["llama_decode", "error", "stat", "llama_free"])
        @test occursin("export llama_decode, llama_free", s)
        @test !occursin(r"export[^\n]*\berror\b", s)   # withheld from the export line
        @test occursin("error, stat", s)               # ...but named in the banner
        @test occursin("Withheld from `export`", s)
    end

    @testset "live: using the module must not shadow the caller" begin
        # Build a module whose API includes a name Base also exports, exactly as
        # a generator would, and drive it through the real emitter.
        body = """
        module _ShadowProbe
        $(W._export_statement(["probe_api", "error"]))
        probe_api() = :ok
        error(x) = :library_error   # the library's own `error`, legitimately API
        end
        """
        m = include_string(@__MODULE__, body)

        @test :probe_api in names(m)            # real API exported
        @test !(:error in names(m))             # collision withheld
        @test isdefined(m, :error)              # ...but still defined
        @test getproperty(m, :error)("x") == :library_error   # ...and reachable

        # The point of all of it: `using` this module leaves Base.error working.
        include_string(@__MODULE__, "using ._ShadowProbe")
        @test probe_api() == :ok                            # export took effect
        @test_throws ErrorException error("must raise, not UndefVarError")
    end
end
