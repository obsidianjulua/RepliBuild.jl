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
        # RepliBuild._rm_tree, not a bare rm: on Windows a file with any handle
        # still open cannot be unlinked (POSIX allows it), so the directory
        # reports ENOTEMPTY and `force=true` does not help — the failure is not
        # about permissions. A real-time virus scanner opening the `.dll` this
        # suite has just built is enough to cause it, which made the very first
        # integration testset fail on a tree it had itself produced seconds
        # earlier. _rm_tree retries with a short backoff.
        ispath(p) && RepliBuild._rm_tree(p)
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

# ── 3. Registry tests — OWNED BY runtests.jl, deliberately not included here ──
# `test_registry.jl` needs no toolchain: it is registry/cache mechanics driven
# entirely through `REPLIBUILD_HOME` → tempdir and `mktempdir()`, and `discover`
# (which scans and writes a TOML, and never invokes a compiler). Verified by
# running it with clang hidden from `PATH` — 6/6.
#
# It used to be included by BOTH suites, which meant no file owned it and it ran
# twice for anyone running both. One suite per file now, enforced by the
# disjointness guard in runtests.jl.

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

# ── 6b. Win64 (Microsoft x64) struct ABI decision table ──────────────────────
# The struct classifier used to be x86-64 SysV only; `classifyWin64Struct` adds
# the second convention behind `AbiTarget`. On Linux the host stays SysV, so the
# rules pinned here are exactly the ones nothing on this machine can execute.
#
# Needs clang, which is why it is here and not in CI. It is a SPECIFICATION
# test: a Win64 callee cannot be loaded or run on Linux, so the oracle is clang
# lowering the same signatures for x86_64-w64-windows-gnu, and what it catches
# is an encoded rule that disagrees with clang. It does NOT prove the lowering
# runs correctly on Windows; only a Windows host does that.
#
# Placed with the other static dialect probes rather than at the end, and the
# ordering is load-bearing: a failing top-level testset aborts the rest of this
# file, and §13 is a standing red, so anything after it does not run in-suite.
# This test needs no JIT, no libJLCS and no built fixture — only clang — so it
# belongs with the cheap checks regardless.
#
# Skips (rather than exits) when clang is absent or cannot target Windows.

include(joinpath(TEST_DIR, "test_win64_abi.jl"))

# ── 7. C-bucket in-process libLLVM pipeline ──────────────────────────────────
# Traces the C link/opt path through Julia's resident libLLVM (default) and the
# external escape hatch ([link] fallback = true), asserting DWARF survives each
# stage — the property the in-process path must not silently break.

include(joinpath(TEST_DIR, "test_c_inprocess.jl"))

# ── 7b/7c/7d. Tier 1 (llvmcall + bitcode slicing) — DELIBERATELY NOT RUN ─────
#
# M1 static promotion, M2 slicing and M3 sliced dispatch used to run here. They
# are unwired on purpose (2026-08-19): Tier 1 is the most experimental part of
# RepliBuild, it is OFF BY DEFAULT (`[wrap.tier1] enable = false`, and every
# Hub config pins `enable_lto = false`), and it ships as a side project rather
# than as a supported tier. Their three testsets rebuild test/slice_test/ once
# each — real compute for a path nothing takes by default.
#
# They are NOT orphaned: `runtests.jl`'s "Every test file is wired into a
# suite" testset carries an explicit `experimental` list naming all three, and
# fails if one is deleted, renamed, or quietly re-wired. A NEW Tier-1 test file
# still fails that guard until someone consciously adds it.
#
# Run them by hand when working on Tier 1 (needs the toolchain):
#     julia --project=. test/test_static_promotion.jl
#     julia --project=. test/test_slicer.jl
#     julia --project=. test/test_tier1_dispatch.jl
#
# Slicer.jl, the promotion pass and the [wrap.tier1] knob all stay shipped and
# unchanged — this removes the SUITE cost, not the feature.

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

# ── 14. DWARF DIE attribution: parameter/context boundaries ──────────────────
# Pins the free_opaque phantom-parameter leak (2026-07-26): a struct member
# defined after the last subprogram was attributed to it as a second parameter,
# emitting a two-argument ccall against a one-argument C function. Drives the
# parser with synthetic readelf dumps — no build required — and covers the
# arity guard that now aborts on any signature the DIE tree does not support.

include(joinpath(TEST_DIR, "test_dwarf_attribution.jl"))

# ── 15. Anonymous struct/union support in the C generator ────────────────────
# Pins the toml_datum_t blob (2026-08-01): an unnamed aggregate DIE was dropped
# on DWARF export and the member referencing it typed as `Any`, which failed
# exact-layout resolution and collapsed the ENTIRE enclosing struct to an
# opaque byte blob — 40 bytes, zero named fields, no way to read a parsed
# value. Covers both shapes (`union {...} u;` and C11 `union {...};`), the
# alignment-carrying opaque region, C11 name injection, and the SysV
# SSE-vs-INTEGER region element-type rule under a live by-value crossing.

include(joinpath(TEST_DIR, "test_anonymous_unions.jl"))

# ── 16. Static inspection of emitted code (RepliBuild.Debug) ─────────────────
# The JIT registers DWARF pointing at the generated MLIR, which is what lets a
# debugger show dialect source inside a thunk. Nothing else in this suite would
# notice if that broke: the thunks keep working, the tests keep passing, and the
# capability quietly disappears. This pins it from the static side — dump the
# emitted object, disassemble it, and require the MLIR to come back interleaved
# with the machine code. Also pins what the DWARF does NOT contain, so the
# emissionKind=Full question is answered by a failing test rather than a memory.

include(joinpath(TEST_DIR, "test_debug_inspection.jl"))

# ── 18. SysConfigGen: generate a library's configure-time headers ────────────
# A project whose headers come from feature detection (`configure_file` over a
# `config.h.in`) cannot be BUILT without this — the compile has nothing to
# include. That makes it a build capability, which is why it moved out of
# RepliBuildTooling and into src/Builder/ (2026-08-26). Tooling is for looking
# at what was produced; this produces something.
#
# Needs cmake, hence devtests. The end-to-end case drives a self-contained
# generated CMakeLists, so no network and no vendored library.

include(joinpath(TEST_DIR, "test_sysconfiggen.jl"))
