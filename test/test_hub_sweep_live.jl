# =============================================================================
# test/tools/hub_sweep.jl, live: a sweep of a Hub made here (2026-11-23)
#
# test_hub_sweep_tool.jl holds the pieces. This runs the tool on three
# packages written into a temporary directory, the three ways a real sweep
# ends:
#
#   good      its source is a git dependency (a repository made here, so no
#             network). The tool rebuilds it; its `test_deep.jl` passes and
#             its `test.jl` builds what is missing, wraps and asserts, as a
#             Hub test script does: clean.
#   redtest   its `test_deep.jl` has a failing test. The new wrapper stays,
#             the verdict is a failure with the totals. Its recipe also asks
#             for a thunks library that the wrap does not produce, which only
#             the sweep's check reports. And its `test.jl` calls `clean`:
#             that script must not be run at all.
#   broken    its source does not compile. It had a wrapper folder before
#             (with a file only this test knows), the rebuild cleans it and
#             the build fails: the saved folder must be back, byte for byte.
#
# And the rule the tool exists to hold (John, 2026-11-23): nothing pulls
# sources that have not changed. `good` is built once before the sweep, so its
# checkout is there; the sweep rebuilds from it, and its `.git` is as that
# first clone left it. What a fetch and a second clone do to a checkout is
# read off real git.
#
# Last, the modes that rebuild nothing: `--verify`, `--resume`, `--restore`.
#
# No real Hub is read. Needs clang (C builds, each in a child Julia) and git.
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

# The user's own git configuration must not decide this fixture: a machine set
# to sign tags refuses a bare `git tag v1` ("no tag message?").
_hsl_git(dir, args...) = run(pipeline(RepliBuild.Tools.tool_cmd(
    `git -C $dir -c user.name=t -c user.email=t@example.invalid -c commit.gpgSign=false -c tag.gpgSign=false $args`);
    stdout = devnull, stderr = devnull))

# `sources` is the recipe's own lines for where the code comes from.
function _hsl_package(hub, name; sources = "", deep = nothing, test = nothing, compile = "")
    dir = joinpath(hub, "packages", name)
    mkpath(dir)
    # As a Hub recipe is written: relative to the package.
    write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "$name"
        root = "."

        $sources

        [compile]
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

_hsl_load(mod) = """
    using Test
    include(joinpath(@__DIR__, "julia", "$mod.jl"))
    """

if requires("a sweep of a Hub made here", :clang, :git)
@testset "hub_sweep, live: three packages, three endings" begin
    hub = mktempdir(); out = joinpath(mktempdir(), "sweep")

    # An upstream for `good`: a repository with one tagged commit.
    upstream = joinpath(mktempdir(), "upstream")
    mkpath(upstream)
    write(joinpath(upstream, "good.c"), "int good_add(int a, int b) { return a + b; }\n")
    _hsl_git(upstream, "init", "--quiet")
    _hsl_git(upstream, "add", "good.c")
    _hsl_git(upstream, "commit", "--quiet", "-m", "one")
    _hsl_git(upstream, "tag", "v1")

    good = _hsl_package(hub, "good";
        sources = "[dependencies.good]\ntype = \"git\"\nurl = \"$(escape_string(upstream))\"\ntag = \"v1\"",
        test = """
            using Test, RepliBuild
            const T = joinpath(@__DIR__, "replibuild.toml")
            @testset "good builds" begin
                lib = RepliBuild.build(T)        # nothing to do when it is built
                @test isfile(lib)
                @test isfile(joinpath(@__DIR__, "julia", "Good.jl"))
            end
            """,
        deep = _hsl_load("Good") * """
            @testset "good deep" begin
                @test Good.good_add(2, 3) == 5
                @test Good.good_add(-1, 1) == 0
            end
            """)
    # `aot_thunks` on a C library compiles nothing, which is the shape of a
    # wrap that could not compile its thunks: it succeeds, with no library.
    redtest = _hsl_package(hub, "redtest";
        compile = "source_files = [\"redtest.c\"]\naot_thunks = true",
        deep = _hsl_load("Redtest") * """
            @testset "redtest deep" begin
                @test Redtest.red_twice(4) == 8
                @test Redtest.red_twice(4) == 9        # wrong on purpose
            end
            """,
        test = """
            using RepliBuild
            RepliBuild.clean(joinpath(@__DIR__, "replibuild.toml"))
            error("a script that cleans was run")
            """)
    write(joinpath(redtest, "redtest.c"), "int red_twice(int a) { return 2 * a; }\n")
    broken = _hsl_package(hub, "broken";
        compile = "source_files = [\"broken.c\"]",
        deep = "error(\"never reached: there is no wrapper\")\n")
    write(joinpath(broken, "broken.c"), "int broken_f(int a) { return a +; }\n")
    mkpath(joinpath(broken, "julia"))
    write(joinpath(broken, "julia", "Broken.jl"), "module Broken\nconst BUILD_GENERATOR = \"0.0.1\"\nend\n")
    write(joinpath(broken, "julia", "keepsake.bin"), "only this test wrote me")

    # `good` has been built before: its sources are on disk when the sweep starts.
    redirect_stdout(devnull) do
        RepliBuild.build(joinpath(good, "replibuild.toml"))
    end
    checkout = joinpath(good, ".replibuild_cache", "deps", "good")
    @test isdir(joinpath(checkout, ".git"))
    clone_config = mtime(joinpath(checkout, ".git", "config"))
    sources_before = _HSL.sweep_sources(good)

    status = redirect_stdout(devnull) do
        _HSL.sweep_main([hub, "--out", out, "--jobs", "3", "--timeout", "900"])
    end
    lines = [split(l, '\t') for l in eachline(joinpath(out, "status.tsv"))]
    function step(pkg, name)
        i = findfirst(l -> l[1] == pkg && l[2] == name, lines)
        return i === nothing ? nothing : lines[i]
    end
    summary = read(joinpath(out, "summary.md"), String)
    version = string(RepliBuild.VERSION)

    @testset "the sweep reports a failure, and says which" begin
        @test status == 1
        @test occursin("3 package(s): 1 clean, 2 with a problem.", summary)
        @test occursin("RepliBuild: $version", read(joinpath(out, "environment.txt"), String))
        for p in ("good", "redtest", "broken")
            @test (p, step(p, "verdict") !== nothing) == (p, true)
        end
    end

    @testset "good: rebuilt by the tool, both scripts run, clean" begin
        @test step("good", "verdict")[3] == "0" && step("good", "verdict")[5] == "ok"
        @test step("good", "rebuild")[3] == "0"
        @test step("good", "test_deep")[5] == "2/2"
        @test step("good", "test")[3] == "0" && step("good", "test")[5] == "2/2"
        @test occursin(Regex("\\| good \\| \\d+ \\| $version \\| kept \\|  \\| test_deep 2/2; test 2/2 \\| ok \\|"), summary)
        @test _HSL.sweep_generators(joinpath(good, "julia")) == [version]
        @test isdir(joinpath(out, "before", "good", "julia"))      # what was there was saved first
    end

    @testset "good: the rebuild used the sources that were there" begin
        # The checkout is as the clone before the sweep left it: no second clone, no fetch.
        @test mtime(joinpath(checkout, ".git", "config")) == clone_config
        @test !isfile(joinpath(checkout, ".git", "FETCH_HEAD"))
        @test _HSL.sweep_sources(good) == sources_before
    end

    @testset "redtest: its failing test is the verdict, its cleaning script never ran" begin
        @test step("redtest", "rebuild")[3] == "0"
        @test step("redtest", "test_deep")[3] == "1" && step("redtest", "test_deep")[5] == "1/2"
        verdict = step("redtest", "verdict")
        @test verdict[3] == "1"
        @test occursin("test_deep: exit 1, 1/2", verdict[5])
        @test isfile(joinpath(redtest, "julia", "Redtest.jl"))      # the new wrapper stays
        @test occursin("| redtest |", summary) && occursin("**FAIL**", summary)
        # The wrap that left no thunks library is named, where the wrap itself said nothing.
        @test occursin("no compiled thunks library (the recipe sets aot_thunks)", verdict[5])
        @test occursin("| MISSING |", summary)
        # A test script that cleans is refused, not run: no step, no log, and the reason.
        @test step("redtest", "test") === nothing
        @test !isfile(joinpath(out, "logs", "redtest.test.log"))
        @test occursin("test.jl calls `clean`: not run (a test script must not clean or pull)", verdict[5])
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

    @testset "what real git does to a checkout: local work keeps it, a fetch or a clone does not" begin
        s0 = _HSL.sweep_sources(good)
        _hsl_git(checkout, "rev-parse", "HEAD")
        _hsl_git(checkout, "checkout", "--quiet", "--force", "v1")
        _hsl_git(checkout, "status", "--porcelain")
        @test _HSL.sweep_sources_verdict(s0, _HSL.sweep_sources(good)) == ("kept", String[])
        sleep(0.05)
        _hsl_git(checkout, "fetch", "--quiet", "--tags", "origin")              # what a re-resolve does
        @test _HSL.sweep_sources_verdict(s0, _HSL.sweep_sources(good)) ==
              ("PULLED", ["sources fetched again though their pin did not change: good"])
        s1 = _HSL.sweep_sources(good)
        mv(checkout, joinpath(mktempdir(), "old"))                 # what a bare `clean` and a build do
        sleep(0.05)
        run(pipeline(RepliBuild.Tools.tool_cmd(`git clone --quiet -- $upstream $checkout`); stdout = devnull, stderr = devnull))
        @test first(_HSL.sweep_sources_verdict(s1, _HSL.sweep_sources(good))) == "PULLED"
    end

    @testset "--verify rebuilds nothing, and still refuses a script that cleans" begin
        wrapper = read(joinpath(redtest, "julia", "Redtest.jl"), String)
        out3 = joinpath(mktempdir(), "verify")
        @test redirect_stdout(devnull) do
            _HSL.sweep_main([hub, "--out", out3, "--verify", "--only", "redtest"])
        end == 1
        v = [split(l, '\t') for l in eachline(joinpath(out3, "status.tsv"))]
        @test [l[2] for l in v] == ["test_deep", "verdict"]
        @test read(joinpath(redtest, "julia", "Redtest.jl"), String) == wrapper
        @test !ispath(joinpath(out3, "before"))                     # nothing was saved: nothing was at risk
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
