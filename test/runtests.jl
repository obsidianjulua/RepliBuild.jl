#!/usr/bin/env julia
# RepliBuild.jl — Registry Test Suite
#
# Lightweight tests for `Pkg.test()` / AutoMerge CI.
# No C++ toolchain required — validates package loading, types, and API surface.
# For full integration tests: julia --project=. test/devtests.jl

using Test
using RepliBuild

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

# ── Ingest-mode unit tests (no C++ toolchain required) ───────────────────────

include(joinpath(@__DIR__, "test_ingest.jl"))

# ── Registry unit tests (no C++ toolchain required) ──────────────────────────

include(joinpath(@__DIR__, "test_registry.jl"))

# ── Varargs @ccall emission regression (no toolchain required) ───────────────

include(joinpath(@__DIR__, "test_varargs_emission.jl"))

# ── User-intent TOML preservation across re-discovery (no toolchain required) ─

include(joinpath(@__DIR__, "test_toml_preservation.jl"))

# ── C-generator policy regressions: cstring_owned, macro-shim visibility,
#    blob param trap, bitfield byte-span (no toolchain required) ──────────────

include(joinpath(@__DIR__, "test_c_generator_policies.jl"))

# ── Macro-shim header-collision guard (library-free fixture; needs clang) ────

include(joinpath(@__DIR__, "test_shim_header_guard.jl"))

# ── Git dependency cache is version-aware (local git upstream; needs git) ────

include(joinpath(@__DIR__, "test_dep_cache.jl"))

# ── DAGDiff module tests (synthetic metadata, no C++ toolchain required) ─────

include(joinpath(@__DIR__, "dag_test", "test_dag_diff.jl"))

# ── Wrapper type-binding guard (no toolchain required) ───────────────────────
# A type used in a foreign-call signature but never declared by the module is
# an UndefVarError at include time — the whole wrapper dies, not one function.
# Library-free traces over the checker plus its refusal-to-write behaviour.

include(joinpath(@__DIR__, "test_wrapper_type_bindings.jl"))

# ── Emitted struct size == DWARF byte_size (no toolchain required) ───────────
# A Tier-2 MEMORY-class struct return is stored straight into the caller's
# `Ref{T}`, so an emitted body one byte too large writes past a live Julia
# object. Measures the emitted type string; needs no MLIR or clang.

include(joinpath(@__DIR__, "test_struct_layout.jl"))

# ── Byte-blob struct setters + the Base-name namespace guard (no toolchain) ──
# Blob structs had accessors in one direction only, so a param struct the
# library builds was read-only — the only path into llama.cpp runs through one.

include(joinpath(@__DIR__, "test_blob_setters.jl"))

# ── Export hygiene (no toolchain) ────────────────────────────────────────────
# The consumer-side half of the Base-name problem. `_assert_base_calls_qualified`
# stops the wrapper breaking ITSELF; this stops it breaking whoever `using`s it,
# which is silent until their first failure path runs.

include(joinpath(@__DIR__, "test_export_hygiene.jl"))

# ── char* return policy + the tier-independence guard (no toolchain) ─────────
# The policy lived on the ccall path only, so the C++ MLIR-dispatch branch —
# which skips that path entirely — handed back bare pointers and silently
# discarded any [wrap.cstring_owned] deallocator. One derivation now, and a
# guard on the write path so a tier cannot decide presentation again.

include(joinpath(@__DIR__, "test_cstring_policy.jl"))

# ── Dispatch + layout introspection (no toolchain) ───────────────────────────
# Twelve Hub consumers hand-rolled `kernel_emits_llvmcall` against a private
# kernel, and four more re-parsed compilation_metadata.json for sizes/offsets.
# The wrapper emits both now; these pin the emitters and execute what they emit.

include(joinpath(@__DIR__, "test_introspection.jl"))

# ── Symbol hygiene (no toolchain) ────────────────────────────────────────────
# Itanium thunk symbols (`_ZTh`/`_ZTv`/`_ZTc`) have no DWARF subprogram, so
# their "class" is inferred as the demangler's phrase ("non-virtual thunk to
# Derived") and neither receiver gate gives them `this`. They shipped as
# exported zero-argument wrappers that SIGSEGV'd on call (2026-08-13).

include(joinpath(@__DIR__, "test_symbol_hygiene.jl"))

# ── Config surface (no toolchain) ────────────────────────────────────────────
# A TOML key that parses and does nothing is worse than an absent one: it is
# documented, users set it, and nothing reads it. `[ingest] extra_link_libs` was
# documented as loading libraries and only ever serialized — the test that
# "covered" it asserted the value round-tripped through the PARSER, which is
# true whether or not the feature exists. Every field must now be consumed or
# explicitly reserved, and the feature is asserted by EXECUTING what is emitted.

include(joinpath(@__DIR__, "test_config_surface.jl"))

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
# prevented it, so: every test_*.jl in test/ must be included by runtests.jl
# or devtests.jl. Adding a file and forgetting to wire it now fails here,
# naming the file.

@testset "Every test file is wired into a suite" begin
    # Collect what the suites actually `include`, NOT every mention of a
    # filename — a comment naming a file (this block names one) must not count
    # as wiring it, or the guard silently passes on the case it exists for.
    function included_basenames(path)
        names = Set{String}()
        for line in eachline(path)
            stripped = lstrip(line)
            (startswith(stripped, '#') || !occursin("include(", stripped)) && continue
            for m in eachmatch(r"\"([^\"]+\.jl)\"", stripped)
                push!(names, basename(m.captures[1]))
            end
        end
        return names
    end

    wired = union(included_basenames(joinpath(@__DIR__, "runtests.jl")),
                  included_basenames(joinpath(@__DIR__, "devtests.jl")))

    # DELIBERATELY UNWIRED — Tier 1 (llvmcall + bitcode slicing) is quarantined.
    # It is the most experimental thing in the repo, it is off by default
    # (`[wrap.tier1] enable = false`, and every Hub config pins
    # `enable_lto = false`), and it ships as a side project rather than as a
    # supported tier. Running its suite means rebuilding slice_test three times
    # for a path nothing takes, so devtests no longer does.
    #
    # This list is the difference between "quarantined" and "silently rotted",
    # which is the exact failure this whole testset exists to prevent —
    # test_tier1_dispatch.jl once shipped wired into neither suite and went
    # unnoticed until an audit. Naming them keeps that impossible: a NEW test
    # file still fails the guard until someone consciously puts it here, and
    # the two asserts below stop this list itself from going stale.
    #
    # To run them (needs the toolchain; each rebuilds test/slice_test/):
    #     julia --project=. test/test_static_promotion.jl   # M1 promotion
    #     julia --project=. test/test_slicer.jl             # M2 slicing
    #     julia --project=. test/test_tier1_dispatch.jl     # M3 dispatch
    # Note they build from test/slice_test/replibuild.toml, which is gitignored
    # and regenerated by nothing — see CLAUDE.md; they are machine-local today.
    experimental = Set([
        "test_static_promotion.jl",
        "test_slicer.jl",
        "test_tier1_dispatch.jl",
    ])

    test_files = sort!(filter(f -> startswith(f, "test_") && endswith(f, ".jl"),
                              readdir(@__DIR__)))
    @test length(test_files) > 10        # readdir sanity — the scan found the suite

    # A quarantined file that no longer exists means the list is stale and is
    # now hiding nothing — fail rather than carry a fiction.
    missing_exempt = filter(f -> !(f in test_files), collect(experimental))
    @test isempty(missing_exempt)

    # …and one that IS wired means the quarantine leaked: devtests would be
    # paying for it while this list claims it does not run.
    leaked = filter(f -> f in wired, collect(experimental))
    @test isempty(leaked)

    unwired = filter(f -> !(f in wired) && !(f in experimental), test_files)
    isempty(unwired) || @warn "Test files included by no suite — they never run" unwired
    @test isempty(unwired)

    # ── ONE SUITE OWNS EACH FILE ─────────────────────────────────────────────
    # "Wired into a suite" was the only rule, so a file could sit in BOTH and
    # satisfy it. `test_registry.jl` did: no suite owned it, it ran twice for
    # anyone running both, and the question "is this a CI test or a toolchain
    # test?" had no answer in the tree — which is the same ambiguity that lets a
    # toolchain dependency drift into CI unnoticed.
    #
    # The split is the contract: runtests.jl is what runs with no C/C++
    # toolchain, devtests.jl is everything that needs one. A file belongs to
    # exactly one of them. If a genuine reason to share ever appears, name the
    # file in SHARED with that reason — the point is that it becomes a decision
    # someone made rather than an accident nobody noticed.
    SHARED = Dict{String,String}()

    ci_files  = included_basenames(joinpath(@__DIR__, "runtests.jl"))
    dev_files = included_basenames(joinpath(@__DIR__, "devtests.jl"))
    overlap   = filter(f -> !haskey(SHARED, f), collect(intersect(ci_files, dev_files)))
    isempty(overlap) || @warn "Included by BOTH suites — give it one owner, or declare it in SHARED with a reason" overlap
    @test isempty(overlap)

    # A stale SHARED entry is a claim about a file that no longer overlaps.
    stale_shared = filter(f -> !(f in intersect(ci_files, dev_files)), collect(keys(SHARED)))
    @test isempty(stale_shared)

    # The scan has to be seeing both suites for any of the above to mean
    # anything — an empty side would make the intersection trivially empty.
    @test length(ci_files)  > 5
    @test length(dev_files) > 5
end
