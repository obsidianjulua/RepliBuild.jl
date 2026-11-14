#!/usr/bin/env julia
# RepliBuild.jl — the pkg suite: what `Pkg.test()` and registration need.
#
# The package loads, VERSION has one source, the config surface, the environment
# doctor, and the structural guards over the test tree (test/suites.jl: every
# test file in exactly one suite, no include-time exit, every import declared).
# Seconds, no C/C++ toolchain.
#
# The toolchain-free regression tests are the unit suite,
#     julia --project=. test/unit.jl            (test/run.jl unit)
# and everything that compiles, links or JITs is the dev suite,
#     julia --project=. test/devtests.jl        (test/run.jl dev)
# both selectable by area or by diff: test/run.jl --help.

using Test
using RepliBuild

# The registry and the Julia environments are the user's. `register` writes to
# ~/.replibuild and adds packages to the default Julia environment, so the suite
# points both somewhere else first; a test that wants either names it.
ENV["REPLIBUILD_HOME"] = mktempdir()
ENV["REPLIBUILD_JULIA_ENV"] = "none"

@testset "RepliBuild.jl" begin

    @testset "Package loads" begin
        # VERSION const must track Project.toml — the pre-3.0.0 three-way
        # version drift started exactly here (hardcoded literals go stale)
        @test RepliBuild.VERSION == pkgversion(RepliBuild)
        @test isdefined(RepliBuild, :discover)
        @test isdefined(RepliBuild, :build)
        @test isdefined(RepliBuild, :wrap)
        @test isdefined(RepliBuild, :clean)
        @test isdefined(RepliBuild, :info)
        @test isdefined(RepliBuild, :check_environment)
        @test isdefined(RepliBuild, :search)
    end

    @testset "ConfigurationManager" begin
        cfg = RepliBuild.ConfigurationManager

        mktempdir() do dir
            toml_path = joinpath(dir, "replibuild.toml")
            write(toml_path, """
            [project]
            name = "test_pkg"
            uuid = "00000000-0000-0000-0000-000000000000"
            root = "."

            [compile]
            source_files = ["src/foo.cpp"]
            include_dirs = ["include"]
            flags = ["-std=c++17"]

            [link]
            enable_lto = false
            optimization_level = "2"

            [binary]
            type = "shared"

            [wrap]
            style = "module"
            """)

            config = cfg.load_config(toml_path)
            @test config.project.name == "test_pkg"
            @test "-std=c++17" in config.compile.flags
            @test config.link.optimization_level == "2"
            @test config.binary.type == :shared
        end
    end

    @testset "DWARFParser types" begin
        dp = RepliBuild.DWARFParser

        vm = dp.VirtualMethod("foo", "_ZN1A3fooEv", 0, "int", String[])
        @test vm.name == "foo"
        @test vm.slot == 0

        ci = dp.ClassInfo("A", 0, String[], [vm], dp.MemberInfo[], 8)
        @test ci.name == "A"
        @test length(ci.virtual_methods) == 1
    end

    @testset "JLCSIRGenerator" begin
        dp = RepliBuild.DWARFParser
        gen = RepliBuild.JLCSIRGenerator

        vm = dp.VirtualMethod("bar", "_ZN1B3barEv", 0, "void", ["int"])
        ci = dp.ClassInfo("B", 0, String[], [vm], dp.MemberInfo[], 8)

        ir = gen.generate_type_info_ir("B", ci, UInt64(0x1000))
        @test contains(ir, "jlcs.type_info")
        @test contains(ir, "\"B\"")

        ir_m = gen.generate_virtual_method_ir(vm, UInt64(0x2000))
        @test contains(ir_m, "thunk__ZN1B3barEv")
    end

    @testset "MLIR native library" begin
        mlir = RepliBuild.MLIRNative
        if isfile(mlir.libJLCS)
            ctx = mlir.create_context()
            @test ctx != C_NULL

            mod = mlir.parse_module(ctx, """
                module { func.func @id(%x: i32) -> i32 { return %x : i32 } }
            """)
            @test mod != C_NULL

            mlir.destroy_context(ctx)
        else
            @info "libJLCS not found — skipping MLIR native tests (Tier 1 still works)"
        end
    end

    @testset "Environment doctor" begin
        status = RepliBuild.check_environment(verbose=false)
        @test hasproperty(status, :ready) || hasfield(typeof(status), :ready)
    end
end

# ── Version is one number, read two ways (no toolchain) ──────────────────────
# `RepliBuild.VERSION` is derived from Project.toml, so this is not tautological:
# `pkgversion` answers from Julia's own package resolution, an independent path.
# Agreement means the file the module read and the file Julia loaded are the same
# one — which is exactly what breaks when a package is dev'd from one checkout
# and loaded from another.
#
# It matters because VERSION feeds `_generator_fingerprint`, which gates the
# registry build cache: a wrong version there serves a stale-codegen wrapper
# instead of rebuilding it.

@testset "Version has a single source of truth" begin
    # Explicit: runtests.jl loads only Test + RepliBuild, and a name that happens
    # to resolve through a dependency today is one refactor from an
    # `UndefVarError` that a try/catch would swallow into a vacuous pass.
    import TOML

    declared = VersionNumber(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])
    @test RepliBuild.VERSION == declared
    @test RepliBuild.VERSION == pkgversion(RepliBuild)

    # A hardcoded literal would satisfy the checks above on the day it was
    # written and drift the next time Project.toml moved. Assert the derivation
    # itself, so re-introducing a literal fails here rather than silently later.
    src = read(joinpath(@__DIR__, "..", "src", "RepliBuild.jl"), String)
    @test occursin("VersionNumber(TOML.parsefile(_PROJECT_TOML)", src)
    @test occursin("include_dependency(_PROJECT_TOML)", src)
    @test !occursin(r"const VERSION = v\"", src)
end

# ── Suite wiring guard (no toolchain required) ───────────────────────────────
# A test file that no suite includes is a test that silently never runs. That
# is exactly what happened to test_tier1_dispatch.jl: it shipped with the M3
# slicing commit wired into neither suite and went unnoticed until an audit
# ran it by hand (2026-07-25, 42/42 on first execution). Nothing structural
# prevented it, so: every test_*.jl in test/ must be listed in test/suites.jl,
# in exactly one suite, or named in its EXPERIMENTAL list. The suites do not
# `include` test files themselves any more (run.jl reads the list), so a stray
# include in a suite file is a second owner and fails here too.

include(joinpath(@__DIR__, "suites.jl"))

@testset "Every test file is wired into exactly one suite" begin
    test_files = sort!(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(@__DIR__)))
    @test length(test_files) > 10        # readdir sanity — the scan found the suite

    listed = [basename(t.file) for t in TESTS]
    # Listed twice is two owners.
    dup = [f for f in unique(listed) if count(==(f), listed) > 1]
    @test isempty(dup)
    # Listed but gone is a claim about nothing; the full relative path must exist.
    missing_listed = [t.file for t in TESTS if !isfile(joinpath(@__DIR__, t.file))]
    @test isempty(missing_listed)

    # DELIBERATELY UNWIRED — Tier 1 (llvmcall + bitcode slicing) is quarantined
    # (CLAUDE.md). This list is the difference between "quarantined" and
    # "silently rotted": a new file still fails below until someone lists it,
    # and the two asserts keep the list itself from going stale.
    missing_exempt = filter(f -> !(f in test_files), EXPERIMENTAL)
    @test isempty(missing_exempt)
    leaked = filter(f -> f in listed, EXPERIMENTAL)
    @test isempty(leaked)

    unwired = filter(f -> !(f in listed) && !(f in EXPERIMENTAL), test_files)
    isempty(unwired) || @warn "Test files in no suite — they never run; list them in test/suites.jl" unwired
    @test isempty(unwired)

    # Every suite has a body, and the areas, locks and `after` edges name things
    # that exist: a fixture lock is a directory under test/, an `after` job is a
    # fixture or a listed file, an area is one `areas_of_path` can answer.
    @test count(t -> t.suite === :unit, TESTS) > 5
    @test count(t -> t.suite === :dev, TESTS) > 5
    @test all(t -> t.suite in (:unit, :dev), TESTS)
    known_areas = Set([:build, :dwarf, :c, :cpp, :dialect, :graph])
    @test all(t -> !isempty(t.areas) && issubset(Set(t.areas), known_areas), TESTS)
    @test all(f -> issubset(Set(f.areas), known_areas), FIXTURES)
    jobs = Set(vcat(["fixture:" * f.name for f in FIXTURES], [t.file for t in TESTS]))
    @test all(t -> all(a -> a in jobs, t.after), TESTS)
    @test all(t -> all(l -> isdir(joinpath(@__DIR__, l)), t.locks), TESTS)
    @test all(f -> isdir(joinpath(@__DIR__, f.name)) && isfile(joinpath(@__DIR__, f.name, "verify.jl")), FIXTURES)
    # A file that runs after a fixture must also lock it, or it can run beside
    # another job rebuilding that directory.
    @test all(t -> all(a -> !startswith(a, "fixture:") || a[9:end] in t.locks, t.after), TESTS)

    # The suite files delegate to run.jl: none includes a test file directly.
    for suite in ("unit.jl", "devtests.jl", "runtests.jl")
        for line in eachline(joinpath(@__DIR__, suite))
            s = lstrip(line)
            (startswith(s, '#') || !occursin("include(", s)) && continue
            @test (suite, line, any(m -> occursin(r"test_[A-Za-z0-9_]*\.jl", m.match), eachmatch(r"\"[^\"]+\"", s))) ==
                  (suite, line, false)
        end
    end

    # The change map answers for every source file, and the whole-tree paths exist.
    @test areas_of_path("src/Builder/GraphMode.jl") == [:graph]
    @test areas_of_path("src/Wrapper/C/GeneratorC.jl") == [:c]
    @test areas_of_path("src/Wrapper/Utils.jl") == [:c, :cpp]
    @test areas_of_path("src/Builder/Compiler.jl") == [:dwarf, :build, :c, :cpp]
    @test areas_of_path("Project.toml") === :all
    @test areas_of_path("docs/src/index.md") == Symbol[]
    for (root, _, files) in walkdir(joinpath(dirname(@__DIR__), "src")), f in files
        endswith(f, ".jl") || continue
        rel = replace(relpath(joinpath(root, f), dirname(@__DIR__)), '\\' => '/')
        a = areas_of_path(rel)
        @test (rel, a === :all || (a isa Vector && !isempty(a))) == (rel, true)
    end
end

# ── No included file can end the suite early ─────────────────────────────────
# `exit(0)` inside an `include` ends the whole run with a SUCCESS status: every
# file after it silently never runs and the suite still reports green. Six
# toolchain files used it as their "libJLCS not built" skip, and
# test_mlir_templates.jl ran first in the libJLCS group, so a machine without
# the dialect skipped most of devtests and passed. They wrap their body in
# `if requires(...)` now (test/support/TestSupport.jl).
#
# Checked on the parsed AST, not the text: a subprocess driver held in a string
# literal (test_jlcs_invariants.jl has two) legitimately calls `exit`, and a
# comment explaining this rule mentions it. What counts is code that runs at
# include time — top level, testset bodies, `if`/`let`/`for`/`try` blocks and
# `do` blocks — not the body of a function, macro or lambda.

@testset "No include-time exit in a suite-included file" begin
    # Full paths, from every quoted segment of an include line:
    # include(joinpath(TEST_DIR, "callback_test", "test_exceptions.jl")).
    function included_paths(suite)
        paths = String[]
        for line in eachline(suite)
            s = lstrip(line)
            (startswith(s, '#') || !occursin("include(", s)) && continue
            parts = [m.captures[1] for m in eachmatch(r"\"([^\"]+)\"", s)]
            (isempty(parts) || !endswith(parts[end], ".jl")) && continue
            push!(paths, joinpath(@__DIR__, parts...))
        end
        return paths
    end

    is_exit(f) = f === :exit || f == :(Base.exit)
    is_short_def(ex) = ex.head === :(=) && ex.args[1] isa Expr &&
        (ex.args[1].head in (:call, :where) ||
         (ex.args[1].head === :(::) && ex.args[1].args[1] isa Expr &&
          ex.args[1].args[1].head === :call))

    # Walks in source order, carrying the last LineNumberNode so a hit names
    # its line. Returns that line.
    function exits!(hits, ex, line)
        ex isa LineNumberNode && return ex.line
        ex isa Expr || return line
        (ex.head in (:function, :macro, :->, :quote) || is_short_def(ex)) && return line
        if ex.head === :do
            # `f() do x … end` runs its lambda now; scan the body, not the head.
            line = exits!(hits, ex.args[1], line)
            return exits!(hits, ex.args[2].args[2], line)
        end
        ex.head === :call && is_exit(ex.args[1]) && push!(hits, line)
        for a in ex.args
            line = exits!(hits, a, line)
        end
        return line
    end

    function include_time_exits(src::AbstractString)
        ast = Meta.parseall(src)
        bad = filter(a -> a isa Expr && a.head in (:incomplete, :error), ast.args)
        isempty(bad) || error("does not parse: $(first(bad))")
        hits = Int[]
        exits!(hits, ast, 0)
        return hits
    end

    # The walker itself, on shapes whose answer is known. Without these the
    # empty result below could just mean the walk never reached anything.
    @test include_time_exits("exit(0)") == [1]
    @test include_time_exits("if !ok\n    @info \"skip\"\n    exit(0)\nend") == [3]
    @test include_time_exits("@testset \"t\" begin\n  Base.exit(1)\nend") == [2]
    @test include_time_exits("mktempdir() do d\n  exit(0)\nend") == [2]
    @test isempty(include_time_exits("f() = exit(0)"))
    @test isempty(include_time_exits("function g()\n exit(1)\nend"))
    @test isempty(include_time_exits("h = x -> exit(x)"))
    @test isempty(include_time_exits("d = \"\"\"\nexit(3)\n\"\"\""))
    @test isempty(include_time_exits("# exit(0)"))

    # Every file a suite runs: the listed test files, the fixture driver and the
    # runner (run.jl's own `exit` is at top level by design, so it is checked
    # for include-time exits in its functions only through the AST walk below;
    # its one top-level call is the entry point and is excluded).
    files = unique(vcat([joinpath(@__DIR__, t.file) for t in TESTS],
                        [joinpath(@__DIR__, "support", "fixture.jl")],
                        included_paths(joinpath(@__DIR__, "runtests.jl"))))
    # A path that does not resolve means the reconstruction is wrong, and a
    # file it cannot read is a file it silently did not check.
    @test all(isfile, files)
    @test length(files) > 40

    offenders = Dict{String,Vector{Int}}()
    for f in filter(isfile, files)
        hits = include_time_exits(read(f, String))
        isempty(hits) || (offenders[relpath(f, @__DIR__)] = hits)
    end
    isempty(offenders) || @warn "Include-time exit — use `if requires(...)` from test/support/TestSupport.jl" offenders
    @test isempty(offenders)
end

# ── Every test import is declared, so `Pkg.test()` works ─────────────────────
# `Pkg.test` builds an environment from [deps] plus the [targets] test extras.
# A test file importing anything outside that set loads fine under
# `julia --project=. test/runtests.jl` — the package's own deps are direct there
# — and dies under `Pkg.test()`, which is how CI invokes it. That is exactly
# what happened: test_struct_layout.jl imports Logging, which was neither a dep
# nor an extra, so `Pkg.test()` ran 21 testsets and then errored with
# "Package Logging not found in current path" while the same suite was green
# the way it gets run by hand.
#
# Checking the declaration rather than shelling out to Pkg.test keeps this cheap
# and keeps the failure legible: it names the package and the file.

@testset "Test imports are declared for Pkg.test" begin
    proj = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    deps   = Set(keys(get(proj, "deps", Dict{String,Any}())))
    extras = Set(keys(get(proj, "extras", Dict{String,Any}())))
    target = Set(get(get(proj, "targets", Dict{String,Any}()), "test", String[]))

    # An extra that is not in the test target is not in the test environment.
    @test issubset(target, extras)

    available = union(deps, target, Set(["RepliBuild"]))

    test_files = filter(f -> startswith(f, "test_") && endswith(f, ".jl"),
                        readdir(@__DIR__))
    @test length(test_files) > 10

    undeclared = Dict{String,Vector{String}}()
    for f in test_files
        for line in eachline(joinpath(@__DIR__, f))
            m = match(r"^\s*(?:using|import)\s+([A-Za-z][A-Za-z0-9_]*)", line)
            m === nothing && continue
            pkg = m.captures[1]
            # `using .Foo` / submodules of the package under test are not deps.
            (pkg in available || startswith(pkg, "RepliBuild")) && continue
            push!(get!(() -> String[], undeclared, pkg), f)
        end
    end

    isempty(undeclared) || @warn "Imported by tests but absent from [deps] and [targets].test — Pkg.test() will fail" undeclared
    @test isempty(undeclared)

    # The scan must actually be finding imports, or the assert above is vacuous
    # — the same "asserting into an empty loop" failure this suite guards
    # elsewhere.
    seen = Set{String}()
    for f in test_files, line in eachline(joinpath(@__DIR__, f))
        m = match(r"^\s*(?:using|import)\s+([A-Za-z][A-Za-z0-9_]*)", line)
        m === nothing || push!(seen, m.captures[1])
    end
    @test "Test" in seen
    @test length(seen) >= 5
end
