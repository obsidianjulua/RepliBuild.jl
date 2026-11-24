# =============================================================================
# test/tools/cpp_type_matrix.jl, the parts that need no compiler (2026-11-23)
#
# The type matrix is a hand-run tool: it builds one small C++ library per
# standard-library type family, wraps it, and runs a probe that calls the
# wrapper from Julia. Its verdicts are read by people deciding what to build
# next, so the bookkeeping must not invent or lose one. Held here:
#
#   * every family is whole (a library, a recipe, a probe that parses and
#     ends), so a family cannot be skipped without anyone seeing;
#   * a probe's lines are read back as written, whatever the library printed
#     around them;
#   * a probe that dies in a call is run again from the case after it, and the
#     case that killed it is the CRASH (the tool and the probe count cases the
#     same way);
#   * an exception is classified once: a refusal, a missing function, or an
#     error;
#   * the table has one row per type, a column per cell, and loses no case.
#
# Nothing is compiled and no wrapper is loaded: the probes here are stand-ins.
# =============================================================================

using Test
using RepliBuild
using TOML

isdefined(@__MODULE__, :TypeMatrixTool) || @eval module TypeMatrixTool
    include(joinpath(@__DIR__, "tools", "cpp_type_matrix.jl"))
end
const _TMX = TypeMatrixTool

# The probe's own support file, in a module of its own: it reads ARGS and
# keeps a counter.
isdefined(@__MODULE__, :TypeMatrixProbe) || @eval module TypeMatrixProbe
    include(joinpath(@__DIR__, "tools", "cpp_type_matrix", "probe_support.jl"))
end
const _TMP = TypeMatrixProbe

# Named at top level: a MethodError spells a function by its name.
_tm_plus1(x::Int) = x + 1

_tm_parses(path) = (ex = Meta.parseall(read(path, String));
                    !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args))

@testset "cpp_type_matrix: the parts that need no compiler" begin

    @testset "every family is whole" begin
        fams = _TMX.tm_families()
        @test length(fams) >= 20
        @test issorted(fams) && allunique(fams)
        checked = 0
        for f in fams
            dir = joinpath(_TMX.TM_DIR, f)
            @test isfile(joinpath(dir, "family.toml"))
            fam = TOML.parsefile(joinpath(dir, "family.toml"))
            @test issubset(keys(fam), ["templates", "template_headers"])
            # A header a recipe names without `<…>` is one of the family's own.
            for h in get(fam, "template_headers", String[])
                startswith(h, "<") || @test isfile(joinpath(dir, h))
            end
            probe = joinpath(dir, "probe.jl")
            @test isfile(probe)
            @test _tm_parses(probe)
            text = read(probe, String)
            @test occursin("probe_support.jl", text)
            @test occursin("load_wrapper()", text)
            # A probe that does not end is read as one that died.
            @test endswith(rstrip(text), "done()")
            # Every case names its type and one cell.
            @test !isempty(collect(eachmatch(r"^case\("m, text)))
            checked += 1
        end
        @test checked == length(fams)
        @test _tm_parses(joinpath(_TMX.TM_DIR, "probe_support.jl"))
    end

    @testset "a family is staged as a project" begin
        hub = mktempdir()
        flags = ["-std=c++20", "-O1"]
        dst = _TMX.tm_stage(hub, "vector", flags)
        @test dst == joinpath(hub, "packages", "tm_vector")
        for f in ("lib.cpp", "types.hpp", "probe.jl", "probe_support.jl", "replibuild.toml")
            @test isfile(joinpath(dst, f))
        end
        recipe = TOML.parsefile(joinpath(dst, "replibuild.toml"))
        @test recipe["project"]["name"] == "tm_vector"
        @test recipe["wrap"]["language"] == "cpp"
        @test recipe["compile"]["flags"] == flags
        @test recipe["compile"]["source_files"] == ["lib.cpp"]
        @test "std::vector<int>" in recipe["types"]["templates"]
        @test "types.hpp" in recipe["types"]["template_headers"]
        # Module names are the wrapper's own derivation, never spelled here.
        @test RepliBuild.ConfigurationManager.wrapper_module_name(joinpath(dst, "replibuild.toml")) == "TmVector"

        # A family with no templates states none.
        plain = _TMX.tm_stage(hub, "complex", flags)
        @test !haskey(TOML.parsefile(joinpath(plain, "replibuild.toml"))["types"], "templates")

        # Staging again leaves an unchanged source alone: a rewritten file
        # would move the project's hash and rebuild for nothing.
        before = mtime(joinpath(dst, "lib.cpp"))
        sleep(1.1)
        _TMX.tm_stage(hub, "vector", flags)
        @test mtime(joinpath(dst, "lib.cpp")) == before
    end

    @testset "a probe's lines are read back as written" begin
        text = """
        some banner
        PROBE_START std::string | make
        v=5
        PROBE std::string | make : WORKS a Julia String is taken
        PROBE_START std::string | pass by value
        PROBE std::string | pass by value : WRONG got 3, the library's answer is 5
        PROBE std::pair<int, double> | takes a Julia Tuple : NOT TAKEN a Tuple: no method pair_sum(::Tuple{Int32, Float64})
        PROBE std::any | return by value : OPAQUE a struct of bytes
        PROBE x | y : MAYBE not a verdict
          PROBE indented | make : WORKS
        PROBE std::function<int(int)> | pass by value : REFUSED cannot call 'fn_apply'
        PROBE a | b : CRASH
        PROBE_DONE
        """
        got = _TMX.tm_parse_probe(text)
        @test [k for (k, _) in got] == [("std::string", "make"), ("std::string", "pass by value"),
                                        ("std::pair<int, double>", "takes a Julia Tuple"), ("std::any", "return by value"),
                                        ("std::function<int(int)>", "pass by value"), ("a", "b")]
        @test [v[1] for (_, v) in got] == ["WORKS", "WRONG", "NOT TAKEN", "OPAQUE", "REFUSED", "CRASH"]
        @test got[1][2][2] == "a Julia String is taken"
        @test got[6][2][2] == ""
        # Every verdict the header documents is one the reader knows.
        for v in _TMX.TM_VERDICTS
            @test length(_TMX.tm_parse_probe("PROBE t | make : $v detail")) == 1
        end
        @test length(_TMX.TM_VERDICTS) == 9
        # "NO WAY" and "NOT TAKEN" both start with "NO": each is read whole.
        @test _TMX.tm_parse_probe("PROBE t | make : NOT TAKEN x")[1][2] == ("NOT TAKEN", "x")
        @test _TMX.tm_parse_probe("PROBE t | make : NO WAY x")[1][2] == ("NO WAY", "x")
    end

    @testset "what a case answers" begin
        @test _TMP.works(3, 3) == "WORKS"
        @test _TMP.works(3, 3, "a note") == "WORKS a note"
        @test startswith(_TMP.works(3, 4), "WRONG got 3, the library's answer is 4")
        @test _TMP.works(0.1 + 0.2, 0.3) == "WORKS"                      # floats: to rounding
        @test startswith(_TMP.works(0.3001, 0.3), "WRONG")
        @test _TMP.works((1, "a"), (1, "a")) == "WORKS"
        @test _TMP.opaque(5, 5, "its bytes") == "OPAQUE its bytes"
        @test startswith(_TMP.opaque(5, 6, "its bytes"), "WRONG")         # a handle that reads wrong is wrong
        @test _TMP.noway("nothing makes one") == "NO WAY nothing makes one"
        # A type is described without the module the wrapper was loaded into.
        @test _TMP._tm_typename("Main.TmAny.any") == "any"
        @test _TMP._tm_typename("Ptr{Main.TmStream.basic_ostream_char}") == "Ptr{basic_ostream_char}"
        @test _TMP._tm_typename("NTuple{8, UInt8}") == "NTuple{8, UInt8}"
        @test _TMP.describe((0x01, 0x02)) == "Tuple{UInt8, UInt8}"

        @test _TMP.takes(() -> _tm_plus1(1), 2, "an Int") == "WORKS an Int"
        @test _TMP.takes(() -> _tm_plus1("s"), 2, "a String") == "NOT TAKEN a String: no method _tm_plus1(::String)"
        @test startswith(_TMP.takes(() -> _tm_plus1(1), 9, "an Int"), "WRONG")
        # Only a MethodError means "not taken": anything else is the case's own.
        @test_throws ErrorException _TMP.takes(() -> error("boom"), 2, "an Int")
    end

    @testset "an exception is classified once" begin
        cls(f) = try f(); "returned" catch e; _TMP._tm_classify(e) end
        @test cls(() -> Base.error("ABI Safety Trap: cannot call 'f' (f(int)). It returns `X` by value")) ==
              "REFUSED cannot call 'f' (f(int)). It returns `X` by value"
        @test cls(() -> throw(UndefVarError(:str_make))) == "ABSENT `str_make` is not in the wrapper"
        @test cls(() -> _tm_plus1("s")) == "ERROR MethodError: no method _tm_plus1(::String)"
        @test cls(() -> error("something else\nsecond line")) == "ERROR something else"
        # A refusal is recognised by its text, whatever throws it.
        @test startswith(cls(() -> throw(ArgumentError("cannot pass a String here"))), "REFUSED")
    end

    @testset "a case prints its start, then its verdict" begin
        seen0 = _TMP._TM_SEEN[]
        out = mktemp() do path, io
            redirect_stdout(io) do
                _TMP.case(() -> _TMP.works(1, 1), "T", "make")
                _TMP.case(() -> error("ABI Safety Trap: cannot call 'f'"), "T", "pass by value")
                _TMP.case(() -> 42, "T", "return by value")           # not a verdict
                _TMP.done()
            end
            flush(io)
            read(path, String)
        end
        @test _TMP._TM_SEEN[] == seen0 + 3
        @test [m.captures[1] for m in eachmatch(r"^PROBE_START (.+)$"m, out)] ==
              ["T | make", "T | pass by value", "T | return by value"]
        got = _TMX.tm_parse_probe(out)
        @test [v[1] for (_, v) in got] == ["WORKS", "REFUSED", "ERROR"]
        @test occursin("not a verdict", got[3][2][2])
        @test occursin(r"^PROBE_DONE$"m, out)
    end

    @testset "a probe that dies is run again from the next case" begin
        # A stand-in probe on the real support file: case 2 ends the process,
        # case 4 never comes back, the rest answer.
        dir = mktempdir()
        cp(joinpath(_TMX.TM_DIR, "probe_support.jl"), joinpath(dir, "probe_support.jl"))
        write(joinpath(dir, "probe.jl"), """
        include(joinpath(@__DIR__, "probe_support.jl"))
        case("A", "make") do; works(1, 1); end
        case("A", "pass by value") do; exit(7); works(1, 1); end
        case("A", "pass by reference") do; works(2, 3); end
        case("B", "make") do; sleep(600); works(1, 1); end
        case("B", "return by value") do; opaque(1, 1, "a handle"); end
        done()
        """)
        logs = mktempdir()
        res = _TMX.tm_run_probe(dir, joinpath(logs, "probe.log"), 15)
        @test [k for (k, _) in res] == [("A", "make"), ("A", "pass by value"), ("A", "pass by reference"),
                                        ("B", "make"), ("B", "return by value")]
        @test [v[1] for (_, v) in res] == ["WORKS", "CRASH", "WRONG", "CRASH", "OPAQUE"]
        @test res[2][2][2] == "exit 7"
        @test res[4][2][2] == "ran out of time"
        # Three runs: to the exit, to the hang, to the end.
        @test length(filter(f -> startswith(f, "probe.log."), readdir(logs))) == 3

        # A probe that cannot start is one line, not silence.
        bad = mktempdir()
        write(joinpath(bad, "probe.jl"), "error(\"no wrapper here\")\n")
        res2 = _TMX.tm_run_probe(bad, joinpath(logs, "bad.log"), 60)
        @test length(res2) == 1
        @test res2[1][1] == ("(probe)", "(load)") && res2[1][2][1] == "ERROR"
        @test occursin("no wrapper here", res2[1][2][2])
    end

    @testset "the table" begin
        results = Dict{String,Any}(
            "string" => [("std::string", "make") => ("WORKS", "a Julia String"),
                         ("std::string", "return by value") => ("WORKS", ""),
                         ("std::string", "return by value") => ("WORKS", "100 characters"),
                         ("std::string", "return by reference") => ("OPAQUE", "a pointer"),
                         ("std::wstring", "takes a Julia String") => ("NOT TAKEN", "a Julia String")],
            "function" => [("std::function<int(int)>", "pass by value") => ("REFUSED", "cannot call"),
                           ("std::function<int(int)>", "pass by reference") => ("NO WAY", "an empty struct"),
                           ("std::function<int(int)>", "return by value") => ("CRASH", "signal 11"),
                           ("std::function<int(int)>", "return by value") => ("WRONG", "got 1")])
        text = _TMX.tm_matrix(results, Dict("variant" => "rebuild: exit 1"), ["-std=c++20"], "coverage text\n")
        lines = split(text, '\n')
        @test occursin("9 cases: 3 works, 1 opaque, 1 not taken, 1 refused, 1 no way, 1 wrong, 1 crash.", text)
        header = only(filter(l -> startswith(l, "| Type |"), lines))
        @test header == "| Type | make | pass by value | pass by reference | return by value | return by reference | other |"
        row(t) = only(filter(l -> startswith(l, "| `$t` |"), lines))
        @test row("std::string") == "| `std::string` | WORKS |  |  | WORKS | OPAQUE |  |"
        @test row("std::wstring") == "| `std::wstring` |  |  |  |  |  | takes a Julia String: NOT TAKEN |"
        # The file says what its words mean.
        for v in _TMX.TM_VERDICTS
            @test occursin("- **" * v * "**", text)
        end
        # Two verdicts in one cell are both shown.
        @test row("std::function<int(int)>") == "| `std::function<int(int)>` |  | REFUSED | NO WAY | CRASH, WRONG |  |  |"
        # Families in order, and every case listed with its detail.
        @test findfirst(==("### function"), lines) < findfirst(==("### string"), lines)
        @test count(l -> startswith(l, "- `"), lines) == 9
        @test occursin("- `std::string`, return by value: **WORKS** — 100 characters", text)
        @test occursin("- **variant**: rebuild: exit 1", text)
        @test occursin("coverage text", text)
        # Nothing built or probed is still a document.
        empty = _TMX.tm_matrix(Dict{String,Any}(), Dict{String,String}(), ["-O1"], "")
        @test occursin("0 cases: .", empty)
    end

    @testset "options" begin
        @test_throws ErrorException _TMX.tm_main(["--no-such-option"])
        @test_throws ErrorException _TMX.tm_main(["--only", "no_such_family", "--out", mktempdir()])
    end
end
