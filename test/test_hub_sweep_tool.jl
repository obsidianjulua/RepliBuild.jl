# =============================================================================
# test/tools/hub_sweep.jl, the parts that need no compiler (2026-11-23)
#
# The sweep tool rebuilds Hub packages in place: it saves each package's
# `julia/` folder, lets a build delete and rewrite it, and puts the saved one
# back when the build gives no wrapper. A mistake there loses a user's
# wrappers, so the pieces are held here on trees made in a temporary
# directory: what is selected, what is read off a test script and a recipe,
# the save and the restore, the step runner's time limit, the totals read off
# a log, and the table. No Hub is read; test_hub_sweep_live.jl runs a sweep.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :HubSweepTool) || @eval module HubSweepTool
    include(joinpath(@__DIR__, "tools", "hub_sweep.jl"))
end
const _HS = HubSweepTool

# A Hub of empty packages: enough for everything that does not build.
function _hs_hub(names...)
    hub = mktempdir()
    for n in names
        mkpath(joinpath(hub, "packages", n))
        write(joinpath(hub, "packages", n, "replibuild.toml"), "[project]\nname = \"$n\"\n")
    end
    mkpath(joinpath(hub, "packages", "not_a_package"))      # no recipe: never a package
    return hub
end

@testset "hub_sweep: the parts that need no compiler" begin

    @testset "options" begin
        o = _HS.sweep_parse_args(["/h", "--out", "/o", "--only", "a, b", "--skip", "c", "--jobs", "3",
                                  "--timeout", "90", "--timeout", "jolt=7200", "--arg", "llamacpp=/m.gguf",
                                  "--arg", "llamacpp=second", "--resume", "--tool-rebuild"])
        @test o.hub == "/h" && o.out == "/o" && o.mode == :rebuild
        @test o.only == ["a", "b"] && o.skip == ["c"] && o.jobs == 3
        @test o.timeout == 90 && o.timeouts == Dict("jolt" => 7200)
        @test o.args == Dict("llamacpp" => ["/m.gguf", "second"])
        @test o.resume && o.tool_rebuild && !o.restore && !o.plan && !o.help
        @test _HS.sweep_parse_args(["--verify"]).mode == :verify
        @test _HS.sweep_parse_args(["--rewrap"]).mode == :rewrap
        @test _HS.sweep_parse_args(["--help"]).help
        @test_throws ErrorException _HS.sweep_parse_args(["--nonsense"])
        @test_throws ErrorException _HS.sweep_parse_args(["/a", "/b"])           # two Hubs
        @test_throws ErrorException _HS.sweep_parse_args(["--arg", "novalue"])
        @test_throws ErrorException _HS.sweep_parse_args(["--jobs", "0"])
        @test_throws ErrorException _HS.sweep_parse_args(["--out"])              # a value is missing
        # The usage text is the file's own header.
        @test occursin("hub_sweep", _HS.sweep_usage()) && occursin("--restore", _HS.sweep_usage())
    end

    @testset "which packages" begin
        hub = _hs_hub("alpha", "beta", "onednn")
        every = _HS.sweep_packages(hub)
        @test every == ["alpha", "beta", "onednn"]
        @test _HS.sweep_packages(joinpath(hub, "packages")) == every          # the packages folder itself
        sel(args...) = _HS.sweep_select(_HS.sweep_parse_args([hub, args...]), every)
        @test sel() == ["alpha", "beta"]                          # oneDNN is left out
        @test sel("--only", "onednn,alpha") == ["alpha", "onednn"]   # unless named
        @test sel("--skip", "alpha") == ["beta"]
        @test sel("--only", "alpha,beta", "--skip", "beta") == ["alpha"]
        # A name that is no package is an error, never an empty sweep.
        @test_throws ErrorException sel("--only", "alhpa")
        @test_throws ErrorException sel("--skip", "not_a_package")
        @test_throws ErrorException _HS.sweep_packages(joinpath(hub, "nowhere"))
    end

    @testset "a test script that rebuilds, read off its code" begin
        dir = mktempdir()
        script(name, text) = (p = joinpath(dir, name); write(p, text); p)
        @test _HS.sweep_rebuilds_itself(script("a.jl", "using RepliBuild\nRepliBuild.clean(T)\nlib = RepliBuild.build(T)\n"))
        @test _HS.sweep_rebuilds_itself(script("b.jl", "using RepliBuild: build\n@testset \"x\" begin\n  lib = build(T)\nend\n"))
        @test _HS.sweep_rebuilds_itself(script("c.jl", "const RB = RepliBuild\nif need\n    RB.clean(T; deps = false)\nend\n"))
        # A verifier: it loads a wrapper and never builds.
        @test !_HS.sweep_rebuilds_itself(script("d.jl", "include(\"julia/X.jl\")\n@test X.f(1) == 2\n"))
        # A mention is not a call.
        @test !_HS.sweep_rebuilds_itself(script("e.jl", "# RepliBuild.clean(T) would delete the clone\nmsg = \"run RepliBuild.build(T) first\"\n"))
        @test !_HS.sweep_rebuilds_itself(script("f.jl", "x = builder(3); cleaned = 1\n"))
        @test !_HS.sweep_rebuilds_itself(joinpath(dir, "absent.jl"))
        @test !_HS.sweep_rebuilds_itself(script("g.jl", "function f(\n"))      # does not parse
    end

    @testset "a recipe that asks for a thunks library" begin
        dir = mktempdir()
        recipe(text) = (p = joinpath(dir, "r$(rand(UInt32)).toml"); write(p, text); p)
        @test _HS.sweep_wants_thunks(recipe("[compile]\naot_thunks = true\n"))
        @test _HS.sweep_wants_thunks(recipe("[wrap]\nlanguage = \"cpp\"\n[platform.linux.compile]\naot_thunks = true\n"))
        @test !_HS.sweep_wants_thunks(recipe("[compile]\naot_thunks = false\n"))
        @test !_HS.sweep_wants_thunks(recipe("[compile]\nflags = [\"-O2\"]\n"))
        @test !_HS.sweep_wants_thunks(recipe("# aot_thunks = true\n[compile]\n"))
        @test !_HS.sweep_wants_thunks(recipe("not toml ["))
    end

    @testset "what a wrapper folder says" begin
        j = joinpath(mktempdir(), "julia")
        mkpath(joinpath(j, "libone"))
        write(joinpath(j, "Top.jl"), "module Top\ninclude(\"libone/One.jl\")\nend\n")           # a graph's index: no stamp
        write(joinpath(j, "libone", "One.jl"), "module One\nconst BUILD_GENERATOR = \"4.6.0\"\nend\n")
        write(joinpath(j, "libone", "libone.so"), "x")
        @test _HS.sweep_generators(j) == ["4.6.0"]
        @test isempty(_HS.sweep_thunks_libraries(j))
        d0 = _HS.sweep_wrapper_digest(j)
        write(joinpath(j, "libone", "libone_thunks.so"), "x")
        @test basename.(_HS.sweep_thunks_libraries(j)) == ["libone_thunks.so"]
        @test _HS.sweep_wrapper_digest(j) == d0                    # a library is not a wrapper
        write(joinpath(j, "libone", "Two.jl"), "module Two\nconst BUILD_GENERATOR = \"4.5.0\"\nend\n")
        @test _HS.sweep_generators(j) == ["4.5.0", "4.6.0"]
        @test _HS.sweep_wrapper_digest(j) != d0                    # a wrapper written is seen
        @test isempty(_HS.sweep_generators(joinpath(j, "absent")))
        @test _HS.sweep_wrapper_digest(joinpath(j, "absent")) == _HS.sweep_wrapper_digest(mktempdir())
    end

    @testset "the saved wrapper folder, and putting it back" begin
        root = mktempdir()
        pkg = joinpath(root, "pkg"); save = joinpath(root, "out", "before", "pkg"); aside = joinpath(root, "out", "failed", "pkg")
        mkpath(joinpath(pkg, "julia", "sub"))
        write(joinpath(pkg, "julia", "Pkg.jl"), "first")
        write(joinpath(pkg, "julia", "sub", "lib.so"), "binary")
        Sys.iswindows() || symlink("Pkg.jl", joinpath(pkg, "julia", "link.jl"))
        @test _HS.sweep_backup!(joinpath(root, "nopkg"), save) == :none
        @test !ispath(save)
        @test _HS.sweep_backup!(pkg, save) == :saved
        @test read(joinpath(save, "julia", "Pkg.jl"), String) == "first"
        @test read(joinpath(save, "julia", "sub", "lib.so"), String) == "binary"
        Sys.iswindows() || @test islink(joinpath(save, "julia", "link.jl"))
        @test !ispath(joinpath(save, "julia.partial"))
        # A second save keeps the first: it is the state from before the sweep.
        write(joinpath(pkg, "julia", "Pkg.jl"), "second")
        @test _HS.sweep_backup!(pkg, save) == :kept
        @test read(joinpath(save, "julia", "Pkg.jl"), String) == "first"

        # A failed build left something else: it is moved aside, the saved one comes back.
        @test _HS.sweep_restore!(pkg, save, aside)
        @test read(joinpath(pkg, "julia", "Pkg.jl"), String) == "first"
        @test read(joinpath(pkg, "julia", "sub", "lib.so"), String) == "binary"
        @test read(joinpath(aside, "julia", "Pkg.jl"), String) == "second"
        @test read(joinpath(save, "julia", "Pkg.jl"), String) == "first"       # the save is still there
        # The build deleted the folder outright (what `clean` does).
        rm(joinpath(pkg, "julia"); recursive = true)
        @test _HS.sweep_restore!(pkg, save, aside)
        @test read(joinpath(pkg, "julia", "Pkg.jl"), String) == "first"
        # A second failure does not overwrite the first one kept aside.
        write(joinpath(pkg, "julia", "Pkg.jl"), "third")
        @test _HS.sweep_restore!(pkg, save, aside)
        kept = filter(startswith("julia"), readdir(aside))
        @test length(kept) == 2
        @test sort([read(joinpath(aside, k, "Pkg.jl"), String) for k in kept]) == ["second", "third"]
        # Nothing saved: nothing is touched.
        @test !_HS.sweep_restore!(pkg, joinpath(root, "out", "before", "other"), aside)
        @test read(joinpath(pkg, "julia", "Pkg.jl"), String) == "first"
    end

    @testset "the totals of a log" begin
        one = "noise\nTest Summary:       | Pass  Total  Time\nClipper2 Deep Tests |  233    233  0.8s\n"
        @test _HS.sweep_test_totals(one) == Dict("Pass" => 233, "Fail" => 0, "Error" => 0, "Broken" => 0, "Total" => 233)
        red = "Test Summary: | Pass  Fail  Error  Broken  Total  Time\nlua deep      |  183     2      1       3    189  2.2s\n" *
              "  inner       |    5     2                   7  0.1s\nERROR: Some tests did not pass\n"
        t = _HS.sweep_test_totals(red)
        @test (t["Pass"], t["Fail"], t["Error"], t["Broken"], t["Total"]) == (183, 2, 1, 3, 189)
        # Two top-level testsets add up; colour codes are not numbers.
        two = one * "\e[1mTest Summary: | \e[32mPass  \e[36mTotal  Time\e[0m\nmore          |    7      7  0.0s\n"
        @test _HS.sweep_test_totals(two)["Total"] == 240
        @test _HS.sweep_test_totals("ERROR: LoadError: UndefVarError\n") === nothing
        @test _HS.sweep_test_totals("") === nothing

        log = tempname()
        write(log, "building\n\e[91mERROR: \e[39mLoadError: Library creation failed: ld: multiple definition of `f'\nStacktrace:\n")
        @test _HS.sweep_first_error(log) == "ERROR: LoadError: Library creation failed: ld: multiple definition of `f'"
        write(log, "all fine\n")
        @test _HS.sweep_first_error(log) == ""
        @test _HS.sweep_first_error(joinpath(tempdir(), "absent-$(rand(UInt32)).log")) == ""
    end

    @testset "a step: its status, its log, its time limit" begin
        dir = mktempdir(); log = joinpath(dir, "step.log")
        julia = Base.julia_cmd()
        (status, secs) = _HS.sweep_run(`$julia --startup-file=no -e 'println("to stdout"); println(stderr, "to stderr"); println(pwd()); println(ENV["REPLIBUILD_JULIA_ENV"])'`, dir, log, 120)
        text = read(log, String)
        @test status == 0 && secs > 0
        @test occursin("to stdout", text) && occursin("to stderr", text)
        @test occursin(basename(dir), text)                       # it ran in the package's folder
        @test occursin("none", text)                              # and registers into no environment
        (status, _) = _HS.sweep_run(`$julia --startup-file=no -e 'exit(3)'`, dir, log, 120)
        @test status == 3
        # A step that outlives its limit is killed, and the log says so.
        t0 = time()
        (status, _) = _HS.sweep_run(`$julia --startup-file=no -e 'sleep(600)'`, dir, log, 2)
        @test status == 124
        @test time() - t0 < 60
        @test occursin("killed after 2 s", read(log, String))
        @test isempty(_HS.SWEEP_RUNNING)                          # nothing is left tracked
    end

    @testset "the plan for a package" begin
        hub = _hs_hub("rebuilder", "verifier", "bare")
        dir(n) = joinpath(hub, "packages", n)
        write(joinpath(dir("rebuilder"), "test.jl"), "RepliBuild.clean(T); RepliBuild.build(T)\n")
        write(joinpath(dir("rebuilder"), "test_deep.jl"), "1\n")
        write(joinpath(dir("rebuilder"), "test_deep_vectors.jl"), "1\n")
        write(joinpath(dir("rebuilder"), "replibuild.toml"), "[compile]\naot_thunks = true\n")
        write(joinpath(dir("verifier"), "test.jl"), "include(\"julia/V.jl\")\n")
        o = _HS.sweep_parse_args([hub])
        p = _HS.sweep_plan(o, dir("rebuilder"))
        @test p.rebuild == "test.jl" && p.deep == ["test_deep.jl", "test_deep_vectors.jl"] && !p.verifier && p.wants_thunks
        p = _HS.sweep_plan(o, dir("verifier"))
        @test p.rebuild == "tool" && isempty(p.deep) && p.verifier && !p.wants_thunks
        p = _HS.sweep_plan(o, dir("bare"))
        @test p.rebuild == "tool" && isempty(p.deep) && !p.verifier
        @test _HS.sweep_plan(_HS.sweep_parse_args([hub, "--tool-rebuild"]), dir("rebuilder")).rebuild == "tool"
        @test _HS.sweep_plan(_HS.sweep_parse_args([hub, "--rewrap"]), dir("rebuilder")).rebuild == "wrap"
        @test _HS.sweep_plan(_HS.sweep_parse_args([hub, "--verify"]), dir("rebuilder")).rebuild == ""
        # --plan runs nothing and writes nothing.
        out = joinpath(mktempdir(), "out")
        @test redirect_stdout(devnull) do
            _HS.sweep_main([hub, "--out", out, "--plan"])
        end == 0
        @test !ispath(out)
    end

    @testset "a package with nothing to run: the checks that need no child" begin
        # --verify on a package with no test script starts no process, so the
        # thunks-library check is seen by itself. It is the one thing that says
        # a wrap fell back to a JIT wrapper (llama.cpp, the 4.6.0 sweep).
        hub = _hs_hub("asks", "had", "plain", "empty")
        dir(n) = joinpath(hub, "packages", n)
        for n in ("asks", "had", "plain")
            mkpath(joinpath(dir(n), "julia"))
            write(joinpath(dir(n), "julia", "W.jl"), "module W\nconst BUILD_GENERATOR = \"4.5.0\"\nend\n")
        end
        write(joinpath(dir("asks"), "replibuild.toml"), "[compile]\naot_thunks = true\n")
        write(joinpath(dir("had"), "julia", "libhad_thunks.so"), "x")
        o = _HS.sweep_parse_args([hub, "--out", joinpath(mktempdir(), "o"), "--verify"])
        seen = Tuple{String,String}[]
        run(n) = _HS.sweep_package(o, n, (pkg, s) -> push!(seen, (pkg, s.name)))
        r = run("asks")
        @test r.thunks == "MISSING" && !_HS.sweep_ok(r)
        @test r.problems == ["no compiled thunks library (the recipe sets aot_thunks)"]
        write(joinpath(dir("asks"), "julia", "libasks_thunks.so"), "x")
        r = run("asks")
        @test r.thunks == "yes" && _HS.sweep_ok(r)
        # A package that has one and no key for it (a Rust crate) is looked at too.
        r = run("had")
        @test r.thunks == "yes" && _HS.sweep_ok(r)
        r = run("plain")
        @test r.thunks == "" && _HS.sweep_ok(r)
        # An older wrapper is not a problem when nothing was rebuilt: it is what is there.
        @test r.generators_before == ["4.5.0"] && r.generators_after == ["4.5.0"]
        r = run("empty")
        @test r.problems == ["no wrapper to verify"]
        @test all(==("note"), last.(seen))                        # no step ran: nothing has a test script
        @test !ispath(joinpath(o.out, "before"))                  # and nothing was saved or touched
    end

    @testset "the table, and what a resumed sweep skips" begin
        ok = _HS.SweepResult("alpha", "tool", [_HS.SweepStep("rebuild", 0, 41.6, ""), _HS.SweepStep("test_deep", 0, 2.0, "12/12")],
                             ["4.5.0"], ["4.6.0"], "yes", String[], false)
        bad = _HS.SweepResult("beta", "test.jl", [_HS.SweepStep("test", 1, 9.0, "")], ["4.5.0"], String[], "",
                              ["test: exit 1 — ERROR: multiple definition", "no wrapper after the rebuild; the saved `julia/` is back"], true)
        text = _HS.sweep_summary([ok, bad], ["onednn"])
        @test occursin("2 package(s): 1 clean, 1 with a problem. Left out: onednn.", text)
        @test occursin("| alpha | tool | 42 | 4.5.0 → 4.6.0 | yes | test_deep 12/12 | ok |", text)
        @test occursin("| beta | test.jl | 9 | 4.5.0 |  |  | **FAIL** |", text)        # put back: the old wrapper's
        fresh = _HS.SweepResult("gamma", "tool", [_HS.SweepStep("rebuild", 0, 5.0, "")], String[], ["4.6.0"], "", String[], false)
        @test occursin("| gamma | tool | 5 | 4.6.0 |  |  | ok |", _HS.sweep_summary([fresh], String[]))
        @test occursin("- **beta**: no wrapper after the rebuild; the saved `julia/` is back", text)
        @test !occursin("## Problems", _HS.sweep_summary([ok], String[]))
        @test _HS.sweep_ok(ok) && !_HS.sweep_ok(bad)

        out = mktempdir()
        @test isempty(_HS.sweep_done(out))
        write(joinpath(out, "status.tsv"), "alpha\trebuild\t0\t41\t\nalpha\tverdict\t0\t44\tok\nbeta\trebuild\t1\t3\t\n")
        @test _HS.sweep_done(out) == Set(["alpha"])              # beta has no verdict: it runs again
    end
end
