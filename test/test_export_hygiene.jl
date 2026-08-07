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

    # ── Exporting a name the module never binds ──────────────────────────────
    #
    # Measured across the Hub before the filter existed: 102 undefined exports
    # in 5 of 18 packages. Two derivations had drifted from the definitions —
    # union accessors screened out of the DEFINITIONS but left in the export
    # list (sqlite 64, lua 12), and enum members exported under the raw C
    # spelling while `@enum` binds the sanitized one (llamacpp 22 `__RLIMIT_*`,
    # zlib 2 `COPY_`/`LEN_`, miniaudio 2 `ma_dr_wav__…`). Three underscore
    # transforms — leading, trailing, interior — one class.

    @testset "defined-name extraction" begin
        body = """
        module Probe
        @enum _rlimit_resource::Cuint begin
            RLIMIT_CPU = 0
            _RLIMIT_NICE = 13
        end
        const LIB_PATH = "x"
        struct Blob; _data::NTuple{4,UInt8}; end
        mutable struct MState; p::Ptr{Cvoid}; end
        abstract type AbsT end
        short_form(x) = x + 1
        function long_form(a, b); return a; end
        COPY = 3
        end
        """
        d = W._defined_names(body)
        # Every binding form the generators emit, including @enum MEMBERS —
        # `@enum T::U begin A = 1 end` binds `A` as surely as a `const` does,
        # and the member is where the emitter's FINAL spelling lives.
        for n in ("_rlimit_resource", "RLIMIT_CPU", "_RLIMIT_NICE", "LIB_PATH",
                  "Blob", "MState", "AbsT", "short_form", "long_form", "COPY")
            @test n in d
        end
        # Not bindings: struct FIELDS and function LOCALS must not be collected,
        # or the filter keeps undefined names and the guard goes blind.
        @test !("_data" in d)
        @test !("p" in d)

        # A docstring written directly above a definition — NO blank line — makes
        # it an argument of `@doc`, not a toplevel `:function`. This is invisible
        # in the source text and it is most of what the generators emit. Missing
        # it drops every DOCUMENTED name while keeping undocumented ones, which
        # is exactly how the first draft of this filter deleted `setproperty`,
        # `g_count` and the bitfield accessors from a real wrapper's export line.
        doc = """
        module D
        \"\"\"Get member `hdr`.\"\"\"
        function documented_fn(s)::Int
            return 1
        end
        \"\"\"A documented short form.\"\"\"
        documented_short(x) = x
        \"\"\"A documented const.\"\"\"
        const DOCUMENTED_CONST = 7
        @inline inlined_fn() = 2
        end
        """
        dd = W._defined_names(doc)
        @test "documented_fn" in dd
        @test "documented_short" in dd
        @test "DOCUMENTED_CONST" in dd
        @test "inlined_fn" in dd          # any macro wrapping a definition

        # Unparseable input yields an empty set rather than throwing —
        # `_assert_wrapper_parses` owns that diagnosis.
        @test isempty(W._defined_names("module Broken\nfunction f(\nend"))

        @test W._exported_names("module M\nexport a, b\nf() = 1\nend") == ["a", "b"]
        @test isempty(W._exported_names("module M\nf() = 1\nend"))
    end

    @testset "export filter drops undefined names" begin
        # Variant B, verbatim: the definition path sanitizes, the export path
        # does not, so the two spellings diverge.
        body = "module P\n@enum R::Cuint begin\n    _RLIMIT_NICE = 13\nend\nfoo() = 1\nend"

        s = W._export_statement(["foo", "__RLIMIT_NICE"], body)
        @test occursin("export foo", s)
        @test !occursin("__RLIMIT_NICE", s)      # never bound → not promised
        @test occursin("_RLIMIT_NICE", W._export_statement(["foo", "_RLIMIT_NICE"], body))

        # Passing no body keeps the old unfiltered behaviour, so every existing
        # caller is unchanged until it opts in.
        @test W._export_statement(["foo", "__RLIMIT_NICE"]) ==
              "export foo, __RLIMIT_NICE\n\n"

        # Filtering composes with the Base-shadowing withhold rather than
        # replacing it: `error` is defined here, so it is withheld (not dropped),
        # while `ghost` is dropped outright.
        body2 = "module P2\nerror(x) = :lib\nprobe() = 1\nend"
        s2 = W._export_statement(["probe", "error", "ghost"], body2)
        @test occursin("export probe", s2)
        @test occursin("Withheld from `export`", s2)   # error → withheld, still defined
        @test !occursin("ghost", s2)                   # ghost → dropped, never defined

        # Everything dropped ⇒ no export line at all, not `export ` with nothing
        # after it (which is a syntax error).
        @test W._export_statement(["ghost1", "ghost2"], body2) == ""
    end

    @testset "guard refuses an undefined export" begin
        G = RepliBuild.Wrapper

        good = "module Good\nexport f\nf() = 1\nend"
        @test G._assert_exports_defined(good, "Good") === nothing

        # The lua shape: an accessor exported but never emitted.
        bad = "module Bad\nexport f, get_Value_gc\nf() = 1\nend"
        @test_throws ErrorException G._assert_exports_defined(bad, "Bad")
        err = try G._assert_exports_defined(bad, "Bad"); "" catch e; sprint(showerror, e) end
        @test occursin("get_Value_gc", err)          # names the offender
        @test !occursin("export f,", err)            # ...and not the innocent one

        # An unparseable body must NOT report every export as undefined —
        # `_assert_wrapper_parses` runs first and gives the real diagnosis.
        @test G._assert_exports_defined("module B\nexport f\nfunction f(\nend", "B") === nothing
    end
end
