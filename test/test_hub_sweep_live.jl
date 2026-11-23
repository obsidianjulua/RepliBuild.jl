# =============================================================================
# test/tools/hub_sweep.jl, live: a sweep of a Hub made here (2026-11-23)
#
# test_hub_sweep_tool.jl holds the pieces. This runs the tool on three
# packages written into a temporary directory, the three ways a real sweep
# ends:
#
#   good      its `test.jl` cleans, builds and wraps (the tool reads that off
#             the script and lets it do the rebuild), and its `test_deep.jl`
#             passes: clean.
#   redtest   rebuilt by the tool; its `test_deep.jl` has a failing test. The
#             new wrapper stays, the verdict is a failure with the totals. Its
#             recipe also asks for a thunks library that the wrap does not
#             produce, which only the sweep's check reports.
#   broken    its source does not compile. It had a wrapper folder before
#             (with a file only this test knows), `clean` deletes it, the
#             build fails: the saved folder must be back, byte for byte.
#
# Then the modes that rebuild nothing: `--verify`, `--resume`, `--restore`.
#
# No real Hub is read. Needs clang (three C builds, each in a child Julia).
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

isdefined(@__MODULE__, :HubSweepTool) || @eval module HubSweepTool
    include(joinpath(@__DIR__, "tools", "hub_sweep.jl"))
end
const _HSL = HubSweepTool

function _hsl_package(hub, name; source, deep = nothing, test = nothing, compile = "")
    dir = joinpath(hub, "packages", name)
    mkpath(dir)
    write(joinpath(dir, "$name.c"), source)
    # As a Hub recipe is written: relative to the package, nothing absolute.
    write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "$name"
        root = "."

        [compile]
        source_files = ["$name.c"]
        $compile

        [link]
        enable_lto = false

        [wrap]
        language = "c"
        """)
    deep === nothing || write(joinpath(dir, "test_deep.jl"), deep)
    test === nothing || write(joinpath(dir, "test.jl"), test)
    return dir
end

_hsl_load(name, mod) = """
    using Test
    include(joinpath(@__DIR__, "julia", "$mod.jl"))
    """

if requires("a sweep of a Hub made here", :clang)
@testset "hub_sweep, live: three packages, three endings" begin
    hub = mktempdir(); out = joinpath(mktempdir(), "sweep")
    good = _hsl_package(hub, "good"; source = "int good_add(int a, int b) { return a + b; }\n",
        test = """
            using Test, RepliBuild
            const T = joinpath(@__DIR__, "replibuild.toml")
            @testset "good builds" begin
                RepliBuild.clean(T)
                lib = RepliBuild.build(T)
                @test isfile(lib)
                @test isfile(RepliBuild.wrap(T))
            end
            """,
        deep = _hsl_load("good", "Good") * """
            @testset "good deep" begin
                @test Good.good_add(2, 3) == 5
                @test Good.good_add(-1, 1) == 0
            end
            """)
    # `aot_thunks` on a C library compiles nothing, which is the shape of a
    # wrap that could not compile its thunks: it succeeds, with no library.
    redtest = _hsl_package(hub, "redtest"; source = "int red_twice(int a) { return 2 * a; }\n",
        compile = "aot_thunks = true",
        deep = _hsl_load("redtest", "Redtest") * """
            @testset "redtest deep" begin
                @test Redtest.red_twice(4) == 8
                @test Redtest.red_twice(4) == 9        # wrong on purpose
            end
            """)
    broken = _hsl_package(hub, "broken"; source = "int broken_f(int a) { return a +; }\n",
        deep = "error(\"never reached: there is no wrapper\")\n")
    mkpath(joinpath(broken, "julia"))
    write(joinpath(broken, "julia", "Broken.jl"), "module Broken\nconst BUILD_GENERATOR = \"0.0.1\"\nend\n")
    write(joinpath(broken, "julia", "keepsake.bin"), "only this test wrote me")

    status = redirect_stdout(devnull) do
        _HSL.sweep_main([hub, "--out", out, "--jobs", "3", "--timeout", "900"])
    end
    lines = [split(l, '\t') for l in eachline(joinpath(out, "status.tsv"))]
    function step(pkg, name)
        i = findfirst(l -> l[1] == pkg && l[2] == name, lines)
        return i === nothing ? nothing : lines[i]
    end
    summary = read(joinpath(out, "summary.md"), String)

    @testset "the sweep reports a failure, and says which" begin
        @test status == 1
        @test occursin("3 package(s): 1 clean, 2 with a problem.", summary)
        @test isfile(joinpath(out, "environment.txt"))
        @test occursin("RepliBuild: $(RepliBuild.VERSION)", read(joinpath(out, "environment.txt"), String))
        for p in ("good", "redtest", "broken")
            @test (p, step(p, "verdict") !== nothing) == (p, true)
        end
    end

    @testset "good: rebuilt by its own test.jl, verified, clean" begin
        @test step("good", "verdict")[3] == "0" && step("good", "verdict")[5] == "ok"
        @test step("good", "test")[3] == "0" && step("good", "test")[5] == "2/2"
        @test step("good", "rebuild") === nothing                 # the tool did not build it again
        @test step("good", "test_deep")[5] == "2/2"
        @test occursin(Regex("\\| good \\| test\\.jl \\| \\d+ \\| $(RepliBuild.VERSION) \\|  \\| test 2/2; test_deep 2/2 \\| ok \\|"), summary)
        @test _HSL.sweep_generators(joinpath(good, "julia")) == [string(RepliBuild.VERSION)]
        @test !ispath(joinpath(out, "before", "good"))             # it had no wrapper before: nothing to save
    end

    @testset "redtest: rebuilt by the tool, its failing test is the verdict" begin
        @test step("redtest", "rebuild")[3] == "0"
        @test step("redtest", "test_deep")[3] == "1" && step("redtest", "test_deep")[5] == "1/2"
        @test step("redtest", "verdict")[3] == "1"
        @test occursin("test_deep: exit 1, 1/2", step("redtest", "verdict")[5])
        @test isfile(joinpath(redtest, "julia", "Redtest.jl"))      # the new wrapper stays
        @test occursin("| redtest | tool |", summary) && occursin("**FAIL**", summary)
        # And the wrap that left no thunks library is named, where the wrap itself said nothing.
        @test occursin("no compiled thunks library (the recipe sets aot_thunks)", step("redtest", "verdict")[5])
        @test occursin("| MISSING |", summary)
    end

    @testset "broken: the build fails, the saved wrapper folder is back" begin
        @test step("broken", "rebuild")[3] != "0"
        @test step("broken", "test_deep") === nothing              # nothing to verify
        @test occursin("no wrapper after the rebuild; the saved `julia/` is back", step("broken", "verdict")[5])
        @test read(joinpath(broken, "julia", "keepsake.bin"), String) == "only this test wrote me"
        @test read(joinpath(broken, "julia", "Broken.jl"), String) == read(joinpath(out, "before", "broken", "julia", "Broken.jl"), String)
        @test occursin("- **broken**: rebuild: exit", summary)
        # The compiler's own complaint is in the step's log.
        @test occursin("broken.c", read(joinpath(out, "logs", "broken.rebuild.log"), String))
    end

    @testset "--verify rebuilds nothing" begin
        stamp = read(joinpath(good, "julia", "Good.jl"), String)
        out2 = joinpath(mktempdir(), "verify")
        @test redirect_stdout(devnull) do
            _HSL.sweep_main([hub, "--out", out2, "--verify", "--only", "good"])
        end == 0
        v = [split(l, '\t') for l in eachline(joinpath(out2, "status.tsv"))]
        @test [l[2] for l in v] == ["test_deep", "verdict"]         # a rebuilding test.jl is not run
        @test read(joinpath(good, "julia", "Good.jl"), String) == stamp
        @test !ispath(joinpath(out2, "before"))
    end

    @testset "--resume skips what has a verdict; --restore puts a saved folder back" begin
        n = length(lines)
        @test redirect_stdout(devnull) do
            _HSL.sweep_main([hub, "--out", out, "--resume"])
        end == 0
        @test countlines(joinpath(out, "status.tsv")) == n          # nothing ran again
        write(joinpath(broken, "julia", "keepsake.bin"), "overwritten")
        @test redirect_stdout(devnull) do
            _HSL.sweep_main([hub, "--out", out, "--restore", "--only", "broken"])
        end == 0
        @test read(joinpath(broken, "julia", "keepsake.bin"), String) == "only this test wrote me"
        @test read(joinpath(out, "replaced", "broken", "julia", "keepsake.bin"), String) == "overwritten"
    end
end
end # requires
