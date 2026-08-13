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
    test_files = sort!(filter(f -> startswith(f, "test_") && endswith(f, ".jl"),
                              readdir(@__DIR__)))
    @test length(test_files) > 10        # readdir sanity — the scan found the suite

    unwired = filter(f -> !(f in wired), test_files)
    isempty(unwired) || @warn "Test files included by no suite — they never run" unwired
    @test isempty(unwired)
end
