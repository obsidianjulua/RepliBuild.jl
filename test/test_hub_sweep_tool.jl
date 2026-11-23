# =============================================================================
# test/tools/hub_sweep.jl, the parts that need no compiler (2026-11-23)
#
# The sweep tool rebuilds Hub packages in place: it saves each package's
# `julia/` folder, lets a build delete and rewrite it, and puts the saved one
# back when the build gives no wrapper. A mistake there loses a user's
# wrappers, so the pieces are held here on trees made in a temporary
# directory: what is selected, what is read off a test script and a recipe,
# the save and the restore, the step runner's time limit, the totals read off
# a log, and the table. And the rule the tool holds for the Hub: a test script
# that calls `clean` is never run, and sources fetched again under an unchanged
# pin are a failure. No Hub is read; test_hub_sweep_live.jl runs a sweep.
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
                                  "--arg", "llamacpp=second", "--resume"])
        @test o.hub == "/h" && o.out == "/o" && o.mode == :rebuild
        @test o.only == ["a", "b"] && o.skip == ["c"] && o.jobs == 3
        @test o.timeout == 90 && o.timeouts == Dict("jolt" => 7200)
        @test o.args == Dict("llamacpp" => ["/m.gguf", "second"])
        @test o.resume && !o.restore && !o.plan && !o.help
        @test_throws ErrorException _HS.sweep_parse_args(["--tool-rebuild"])     # gone: the tool always rebuilds
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

    @testset "a test script that cleans, read off its code" begin
        dir = mktempdir()
        script(name, text) = (p = joinpath(dir, name); write(p, text); p)
        @test _HS.sweep_cleans(script("a.jl", "using RepliBuild\nRepliBuild.clean(T)\nlib = RepliBuild.build(T)\n"))
        # Keeping the sources is still a clean: the rule is that a test script does not.
        @test _HS.sweep_cleans(script("b.jl", "const RB = RepliBuild\nif need\n    RB.clean(T; deps = false)\nend\n"))
        @test _HS.sweep_cleans(script("c.jl", "using RepliBuild: clean\n@testset \"x\" begin\n  clean(T)\nend\n"))
        # A script that builds what is missing, wraps and asserts is what a test script is.
        @test !_HS.sweep_cleans(script("d.jl", "lib = RepliBuild.build(T)\n@test isfile(RepliBuild.wrap(T))\n"))
        @test !_HS.sweep_cleans(script("e.jl", "include(\"julia/X.jl\")\n@test X.f(1) == 2\n"))
        # A mention is not a call.
        @test !_HS.sweep_cleans(script("f.jl", "# RepliBuild.clean(T) would delete the clone\nmsg = \"run RepliBuild.clean(T) first\"\n"))
        @test !_HS.sweep_cleans(script("g.jl", "x = cleaner(3); cleaned = 1; C.curl_easy_cleanup(h)\n"))
        @test !_HS.sweep_cleans(joinpath(dir, "absent.jl"))
        @test !_HS.sweep_cleans(script("h.jl", "function f(\n"))      # does not parse
    end

    @testset "sources: kept, new, or pulled" begin
        pkg = mktempdir(); deps = joinpath(pkg, ".replibuild_cache", "deps")
        @test isempty(_HS.sweep_sources(pkg))                     # never built: no checkout
        @test _HS.sweep_sources_verdict(Dict(), Dict()) == ("", String[])
        mkpath(joinpath(deps, "lib", ".git"))
        write(joinpath(deps, "lib", ".git", "config"), "[core]\n")
        write(joinpath(deps, "lib.resolved"), "url=u\ntag=v1\ncommit=abc\n")
        mkpath(joinpath(deps, "tarball"))                          # an unpacked archive: no .git
        write(joinpath(deps, "tarball.resolved"), "url=t\nsha256=1\n")
        s0 = _HS.sweep_sources(pkg)
        @test sort!(collect(keys(s0))) == ["lib", "tarball"] && s0["lib"][1] == "url=u\ntag=v1\ncommit=abc\n"
        # Nothing happened: kept.
        @test _HS.sweep_sources(pkg) == s0
        @test _HS.sweep_sources_verdict(s0, s0) == ("kept", String[])
        # A checkout that was not there before is new sources, which is when fetching is right.
        @test _HS.sweep_sources_verdict(Dict{String,Tuple{String,String}}(), s0) == ("fetched (new)", String[])
        # A fetch rewrites FETCH_HEAD: the pin is the same, so this is a pull.
        write(joinpath(deps, "lib", ".git", "FETCH_HEAD"), "abc\n")
        s1 = _HS.sweep_sources(pkg)
        @test s1["lib"][2] != s0["lib"][2] && s1["tarball"] == s0["tarball"]
        @test _HS.sweep_sources_verdict(s0, s1) ==
              ("PULLED", ["sources fetched again though their pin did not change: lib"])
        # The same fetch with a new pin in the marker is a version move.
        write(joinpath(deps, "lib.resolved"), "url=u\ntag=v2\ncommit=def\n")
        @test _HS.sweep_sources_verdict(s0, _HS.sweep_sources(pkg)) == ("new pin", String[])
        # A clone made again (what a bare `clean` leads to) under the old pin: pulled.
        write(joinpath(deps, "lib.resolved"), "url=u\ntag=v1\ncommit=abc\n")
        mv(joinpath(deps, "lib"), joinpath(pkg, "old"))            # the old one still holds its inode
        mkpath(joinpath(deps, "lib", ".git"))
        sleep(0.05)
        write(joinpath(deps, "lib", ".git", "config"), "[core]\n")
        @test first(_HS.sweep_sources_verdict(s0, _HS.sweep_sources(pkg))) == "PULLED"
        # An archive unpacked again, the same way.
        mv(joinpath(deps, "tarball"), joinpath(pkg, "oldtar"))
        mkpath(joinpath(deps, "tarball"))
        (cell, problems) = _HS.sweep_sources_verdict(s0, _HS.sweep_sources(pkg))
        @test cell == "PULLED" && problems == ["sources fetched again though their pin did not change: lib, tarball"]
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
        hub = _hs_hub("full", "cleaner", "bare")
        dir(n) = joinpath(hub, "packages", n)
        write(joinpath(dir("full"), "test.jl"), "lib = RepliBuild.build(T)\n")
        write(joinpath(dir("full"), "test_deep.jl"), "1\n")
        write(joinpath(dir("full"), "test_deep_vectors.jl"), "1\n")
        write(joinpath(dir("full"), "replibuild.toml"), "[compile]\naot_thunks = true\n")
        write(joinpath(dir("cleaner"), "test.jl"), "RepliBuild.clean(T); RepliBuild.build(T)\n")
        write(joinpath(dir("cleaner"), "test_deep.jl"), "1\n")
        o = _HS.sweep_parse_args([hub])
        p = _HS.sweep_plan(o, dir("full"))
        @test p.rebuild == "rebuild" && p.wants_thunks && isempty(p.refused)
        @test p.scripts == ["test_deep.jl", "test_deep_vectors.jl", "test.jl"]     # the deep ones first
        p = _HS.sweep_plan(o, dir("cleaner"))
        @test p.scripts == ["test_deep.jl"] && p.refused == ["test.jl"] && !p.wants_thunks
        p = _HS.sweep_plan(o, dir("bare"))
        @test p.rebuild == "rebuild" && isempty(p.scripts) && isempty(p.refused)
        @test _HS.sweep_plan(_HS.sweep_parse_args([hub, "--rewrap"]), dir("full")).rebuild == "wrap"
        @test _HS.sweep_plan(_HS.sweep_parse_args([hub, "--verify"]), dir("full")).rebuild == ""
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
        # A script that cleans is refused before anything runs, and the package fails for it.
        write(joinpath(dir("plain"), "test.jl"), "RepliBuild.clean(T); error(\"a cleaning script was run\")\n")
        write(joinpath(dir("plain"), "test_deep.jl"), "RepliBuild.clean(T; deps = false); error(\"so was this one\")\n")
        empty!(seen)
        r = run("plain")
        @test r.problems == ["test_deep.jl calls `clean`: not run (a test script must not clean or pull)",
                             "test.jl calls `clean`: not run (a test script must not clean or pull)"]
        @test isempty(r.steps) && isempty(seen)
        @test isfile(joinpath(dir("plain"), "julia", "W.jl"))     # and nothing was cleaned
        @test !ispath(joinpath(o.out, "before"))                  # and nothing was saved or touched
    end

    @testset "the table, and what a resumed sweep skips" begin
        ok = _HS.SweepResult("alpha", "rebuild", [_HS.SweepStep("rebuild", 0, 41.6, ""), _HS.SweepStep("test_deep", 0, 2.0, "12/12")],
                             ["4.5.0"], ["4.6.0"], "yes", "kept", String[], false)
        bad = _HS.SweepResult("beta", "rebuild", [_HS.SweepStep("rebuild", 1, 9.0, "")], ["4.5.0"], String[], "", "kept",
                              ["rebuild: exit 1 — ERROR: multiple definition", "no wrapper after the rebuild; the saved `julia/` is back"], true)
        text = _HS.sweep_summary([ok, bad], ["onednn"])
        @test occursin("2 package(s): 1 clean, 1 with a problem. Left out: onednn.", text)
        @test occursin("| alpha | 42 | 4.5.0 → 4.6.0 | kept | yes | test_deep 12/12 | ok |", text)
        @test occursin("| beta | 9 | 4.5.0 | kept |  |  | **FAIL** |", text)             # put back: the old wrapper's
        fresh = _HS.SweepResult("gamma", "rebuild", [_HS.SweepStep("rebuild", 0, 5.0, "")], String[], ["4.6.0"], "", "fetched (new)", String[], false)
        @test occursin("| gamma | 5 | 4.6.0 | fetched (new) |  |  | ok |", _HS.sweep_summary([fresh], String[]))
        @test occursin("- **beta**: no wrapper after the rebuild; the saved `julia/` is back", text)
        @test !occursin("## Problems", _HS.sweep_summary([ok], String[]))
        @test _HS.sweep_ok(ok) && !_HS.sweep_ok(bad)

        out = mktempdir()
        @test isempty(_HS.sweep_done(out))
        write(joinpath(out, "status.tsv"), "alpha\trebuild\t0\t41\t\nalpha\tverdict\t0\t44\tok\nbeta\trebuild\t1\t3\t\n")
        @test _HS.sweep_done(out) == Set(["alpha"])              # beta has no verdict: it runs again
    end
end
