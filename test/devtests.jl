#!/usr/bin/env julia
# RepliBuild.jl — Full Integration Test Suite
#
# Runs the complete pipeline for each test project: discover → build → wrap → verify.
# Requires: LLVM 21+, Clang, CMake on the system.
# Usage:  julia --project=. test/devtests.jl

using Test
using TOML
using RepliBuild

const TEST_DIR = @__DIR__

# ── Helpers ──────────────────────────────────────────────────────────────────

function clean_test_dir(dir::String)
    for name in ["build", "julia", ".replibuild_cache"]
        p = joinpath(dir, name)
        ispath(p) && rm(p, recursive=true, force=true)
    end
end

"""Run a verify.jl script in a subprocess to avoid module name collisions."""
function run_verify(dir::String; label::String=basename(dir))
    verify = joinpath(dir, "verify.jl")
    isfile(verify) || error("verify.jl not found in $dir")

    project_root = dirname(TEST_DIR)
    cmd = `$(Base.julia_cmd()) --project=$project_root $verify`
    result = run(ignorestatus(cmd))
    return success(result)
end

# ── 1. Pipeline ──────────────────────────────────────────────────────────────

@testset "Pipeline (discover → build → wrap)" begin
    dir = joinpath(TEST_DIR, "stress_test")
    @test isdir(dir)

    # Step-by-step pipeline
    clean_test_dir(dir)

    toml_path = RepliBuild.discover(dir, force=true)
    @test isfile(toml_path)

    library_path = RepliBuild.build(toml_path)
    @test isfile(library_path)
    @test isfile(joinpath(dir, "julia", "compilation_metadata.json"))

    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)
    @test endswith(wrapper_path, ".jl")

    # Info / clean round-trip
    @test_nowarn RepliBuild.info(toml_path)
    RepliBuild.clean(toml_path)
    @test !isdir(joinpath(dir, "build"))
    @test !isdir(joinpath(dir, "julia"))

    # Chained pipeline
    toml_path = RepliBuild.discover(dir, force=true, build=true, wrap=true)
    @test isfile(toml_path)
    julia_dir = joinpath(dir, "julia")
    @test any(endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll") for f in readdir(julia_dir))
    @test any(endswith(f, ".jl") for f in readdir(julia_dir))
end

# ── 2. Integration tests (each in subprocess) ────────────────────────────────

INTEGRATION_TESTS = [
    ("stress_test",        "Stress Test (numerics, vtable, RAII, MLIR)"),
    ("mi_test",            "Multiple Inheritance (two-base layout, upcasts)"),
    ("vi_test",            "Virtual Inheritance (diamond, dynamic vbase upcasts)"),
    ("stl_test",           "STL Templates (vector, string, map)"),
    ("c_test",             "C Fundamentals (structs, enums, LTO, packed, unions)"),
    ("c_abomination_test", "C Edge Cases (opaque structs, nested callbacks)"),
    ("callback_test",      "Callbacks (Julia ↔ C++)"),
]

# Curated fixture config that discovery cannot derive from source (see
# docs/updates/2026-07-17-stl-test-regression.md). Applied after every
# regeneration: discover's user-intent preservation keeps these alive on
# subsequent runs, but the tomls are gitignored — a fresh clone has nothing
# to preserve, so the suite seeds them explicitly. Machine-independent
# values only (no absolute paths).
const CURATED_FIXTURE_CONFIG = Dict(
    "stl_test" => Dict(
        "types" => Dict(
            "templates"        => ["std::vector<int>", "std::string", "std::map<int, int>"],
            "template_headers" => ["<vector>", "<string>", "<map>"],
        ),
    ),
)

function apply_curated_config(toml_path::String, name::String)
    haskey(CURATED_FIXTURE_CONFIG, name) || return
    doc = TOML.parsefile(toml_path)
    for (sec, kv) in CURATED_FIXTURE_CONFIG[name], (k, v) in kv
        get!(doc, sec, Dict{String,Any}())[k] = v
    end
    open(toml_path, "w") do io
        TOML.print(io, doc)
    end
end

const _SKIP = Set(split(get(ENV, "REPLIBUILD_SKIP_TESTS", ""), ',', keepempty=false))

for (name, label) in INTEGRATION_TESTS
    if name in _SKIP
        @info "Skipping $name (REPLIBUILD_SKIP_TESTS)"
        continue
    end
    @testset "$label" begin
        dir = joinpath(TEST_DIR, name)
        @test isdir(dir)

        clean_test_dir(dir)
        # Always regenerate the toml via discover so the suite never depends on a
        # committed config carrying machine-specific absolute paths. The fixture
        # tomls are gitignored for this reason. (Hub packages are different — their
        # tomls are hand-rolled and must NOT be discovered.) Curated, source-
        # underivable sections are seeded AFTER discovery, BEFORE build — this is
        # what regressed stl_test for six weeks when discover(force) silently
        # destroyed [types].templates (2026-07-17).
        toml = RepliBuild.discover(dir, force=true)
        @test isfile(toml)
        apply_curated_config(toml, name)
        RepliBuild.build(toml)
        RepliBuild.wrap(toml)
        @test isdir(joinpath(dir, "julia"))

        @test run_verify(dir; label=label)
    end
end

# ── Ingest mode (BYOB) end-to-end ────────────────────────────────────────────
# Build a fixture, then point ingest at the resulting .so and confirm the wrapper
# generates correctly without re-running the compile pipeline.

@testset "Ingest mode (BYOB)" begin
    src_dir = joinpath(TEST_DIR, "c_test")
    @test isdir(src_dir)

    # Source-build c_test to produce libc_test.so with DWARF
    clean_test_dir(src_dir)
    src_toml = joinpath(src_dir, "replibuild.toml")
    @test isfile(src_toml)
    src_lib = RepliBuild.build(src_toml)
    @test isfile(src_lib)

    # Now ingest that .so into a brand-new project that has no source whatsoever
    mktempdir() do ingest_dir
        toml = RepliBuild.ingest(
            src_lib,
            headers=[joinpath(src_dir, "include")],
            name="c_ingest_e2e",
            project_dir=ingest_dir,
            language=:c,
            register=false,
        )
        @test isfile(toml)

        ingested_lib = RepliBuild.build(toml)
        @test isfile(ingested_lib)
        @test isfile(joinpath(ingest_dir, "julia", "compilation_metadata.json"))

        wrapper = RepliBuild.wrap(toml)
        @test isfile(wrapper)
        @test endswith(wrapper, ".jl")
    end
end

# ── 3. Registry tests ────────────────────────────────────────────────────────

include(joinpath(TEST_DIR, "test_registry.jl"))

# ── 4. MLIR JLCS dialect template stress tests ───────────────────────────────
# Self-skips if libJLCS isn't built; otherwise exercises nested CStructs,
# packed template structs, packed sret returns, RAII ordering, virtual
# dispatch on template containers, TypeInfoOp inheritance, etc.

include(joinpath(TEST_DIR, "test_mlir_templates.jl"))

# ── 5. C++ exception handling through JLCS try_call ──────────────────────────
# Depends on callback_test/julia/CallbackTest.jl, which the integration tests
# above already produce via build+wrap.

include(joinpath(TEST_DIR, "callback_test", "test_exceptions.jl"))

# ── 6. JLCS dialect invariant probes ─────────────────────────────────────────
# Definitive-trace probes that push specific dialect concerns (op arity
# invariants, dead-producer ops) through parse → lower → emit and record the
# actual outcome. Self-skips without libJLCS. Two @test_broken entries mark
# confirmed lowering crashes awaiting verifiers (jlcs.scope, jlcs.marshal_arg).

include(joinpath(TEST_DIR, "test_jlcs_invariants.jl"))

# ── 7. C-bucket in-process libLLVM pipeline ──────────────────────────────────
# Traces the C link/opt path through Julia's resident libLLVM (default) and the
# external escape hatch ([link] fallback = true), asserting DWARF survives each
# stage — the property the in-process path must not silently break.

include(joinpath(TEST_DIR, "test_c_inprocess.jl"))

# ── 7b. Static promotion (llvmcall slicing contract, M1) ─────────────────────
# Fixture-gated: internal statics (functions + mutable globals) become exported
# __rb_<lib>_* symbols post-opt so per-function slices bind to the .so's single
# copy of state; const statics stay internal; the wrapper never wraps __rb_*.

include(joinpath(TEST_DIR, "test_static_promotion.jl"))

# ── 7c. Per-function bitcode slicing (llvmcall slicing M2) ───────────────────
# Slicer.jl over the promoted slice_test module: declarations-only slices
# (single definition, mutable statics declared via __rb_*), hazard/refusal
# policy, both state-coherence directions through live Base.llvmcall, cache.

include(joinpath(TEST_DIR, "test_slicer.jl"))

# ── 8. Nested-struct ABI resolution (pure ccall path) ────────────────────────
# Library-free trace: structs with struct-typed members must come out with
# verified named fields (SysV register classes preserved by value), and
# unreproducible layouts (packed floats) must refuse by-value crossings loudly.

include(joinpath(TEST_DIR, "test_abi_nested.jl"))

# ── 9. Per-file IR cache invalidation on compile-config change ────────────────
# A compile-flag change must invalidate the per-file IR cache without a manual
# `rm -rf` — otherwise stale IR (built with the old flags) is silently reused.

include(joinpath(TEST_DIR, "test_cache_invalidation.jl"))

# ── 10. Convenience-overload ownership guard ─────────────────────────────────
# Library-free trace: no struct-by-value convenience overloads may be emitted
# for Ptr{Struct} params (Ref(copy) handed to a function that frees/retains
# the pointer is UB — crash-proven double-free on cJSON_Delete). The Vector
# input-array path survives, with Cstring returns aligned to the base
# wrapper's String policy.

include(joinpath(TEST_DIR, "test_convenience_overloads.jl"))

# ── 11. JLCS producers: scope-RAII + array-view ──────────────────────────────
# DWARF-driven producers for the previously producer-less op families:
# type_info destructorName, jlcs.scope/ctor_call/dtor_call around by-value
# non-trivial class params (Itanium caller-owned temporary — the old raw-bits
# pass was a miscompile), and zero-copy strided accessors via
# load/store_array_element. Executes through the real MLIR JIT.

include(joinpath(TEST_DIR, "test_jlcs_producers.jl"))

# ── 12. Struct ABI traces: nested c_struct + SysV small-struct returns ───────
# Pins the pugixml JIT-init segfault fixes (2026-07-18): packed structs nested
# in llvm.struct bodies inline as LLVM literals; create_jit refuses foreign
# types catchably; register-class (≤16B aligned) struct returns/args coerce
# one scalar per eightbyte — verified against a real clang++-compiled callee.

include(joinpath(TEST_DIR, "test_struct_abi.jl"))

# ── 13. Multi-library JIT: one engine per wrapped binary ─────────────────────
# Two generated wrappers in one session. The old single-engine singleton let
# the first library win and silently killed the second's Tier 2 (found live
# composing box2d + pugixml). Runs AFTER mi_test/vi_test so their wrappers
# exist; loads both and dispatches Tier-2 thunks through each engine.

include(joinpath(TEST_DIR, "test_multilib_jit.jl"))
