#!/usr/bin/env julia
# RepliBuild.jl — Full Integration Test Suite
#
# Runs the complete pipeline for each test project: discover → build → wrap → verify.
# Requires: LLVM 21+, Clang, CMake on the system.
# Usage:  julia --project=. test/devtests.jl

using Test
using TOML
using RepliBuild

# The registry and the Julia environments are the user's. The chained
# `discover(...; build=true, wrap=true)` below ends in `register`, which writes
# to ~/.replibuild and adds a package to the default Julia environment; every
# devtests run used to register its fixtures in the real registry.
ENV["REPLIBUILD_HOME"] = mktempdir()
ENV["REPLIBUILD_JULIA_ENV"] = "none"

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

    # It ends in `register`: the build made in the project is vendored (to this
    # suite's temp REPLIBUILD_HOME) under the hash `use` will look for.
    cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
    entry = RepliBuild.PackageRegistry._load_index().entries[cfg.project.name]
    @test entry.toml_path == abspath(toml_path)
    @test isfile(joinpath(RepliBuild.PackageRegistry._cached_build_dir(entry.build_hash),
                          RepliBuild.ConfigurationManager.get_module_name(cfg) * ".jl"))
end

# ── 1b. Local-first registry: vendor, follow edits, load by `using` ──────────

include(joinpath(TEST_DIR, "test_registry_local_first.jl"))

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

    # `build=true, wrap=true` registers after wrapping, as discover does: the
    # entry carries a vendored build.
    mktempdir() do tmp
        withenv("REPLIBUILD_HOME" => joinpath(tmp, "rb_home"), "REPLIBUILD_JULIA_ENV" => "none") do
            toml = RepliBuild.ingest(src_lib, headers=[joinpath(src_dir, "include")],
                                     name="c_ingest_reg", project_dir=joinpath(tmp, "proj"),
                                     language=:c, build=true, wrap=true)
            entry = RepliBuild.PackageRegistry._load_index().entries["c_ingest_reg"]
            @test entry.toml_path == abspath(toml)
            @test !isempty(entry.build_hash)
            @test isfile(joinpath(tmp, "rb_home", "builds", entry.build_hash, entry.module_name * ".jl"))
            @test isempty(entry.julia_env)
        end
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

# ── 5b. JIT unwind after a prior engine was destroyed ────────────────────────
# Self-contained (tiny clang++ fixture). Pins the Windows COFF bug where
# RuntimeDyld fed .pdata to __register_frame and teardown poisoned libunwind
# so the next throw through a live JIT frame aborted with "pc not in table".
include(joinpath(TEST_DIR, "test_jit_unwind.jl"))

# ── 5c. AOT thunks end to end ────────────────────────────────────────────────
# `[compile] aot_thunks = true` through discover → build → wrap on its own C++
# fixture (test/aot_test/), cold and again: the first wrap must compile exactly
# the thunks it binds. Calls
# through the companion `_thunks` library in a child process: member, by-value
# records, virtual override, and a C++ throw for every return shape. Placed
# with §5/§5b, which drive the same thunks through the JIT.
include(joinpath(TEST_DIR, "test_aot_thunks.jl"))

# ── 6. JLCS dialect invariant probes ─────────────────────────────────────────
# Definitive-trace probes that push specific dialect concerns (op arity
# invariants, dead-producer ops) through parse → lower → emit. Self-skips
# without libJLCS. The malformed jlcs.scope / jlcs.marshal_arg cases used to
# crash lowering; their verifiers (2026-07-16) reject them at parse, and that
# is what the file asserts.

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

# ── 6c. Wrap-surface reachability guard ──────────────────────────────────────
# Metadata is correlated against `nm -g` (the full symbol table); the generated
# wrapper resolves through `dlsym`, i.e. `nm -D`. Nothing checked that the two
# agree, and `-fvisibility=hidden` × `[link] promote_statics` is what makes them
# disagree — hidden reads as "internal", promotion renames the definition to
# `__rb_*`, and a library whose export macro is a bare `extern` (lua's LUA_API)
# has its whole API renamed out of the wrap surface by a build that succeeded.
#
# Placed here with the other cheap probes, and for the same reason §6b is: it
# needs only the JLL clang and `nm` — no JIT, no libJLCS, no system LLVM, no
# built fixture — and a failing top-level testset aborts everything after it, so
# cheap checks must not sit downstream of heavy ones.

include(joinpath(TEST_DIR, "test_wrap_surface_guard.jl"))

# ── 6d. Per-library export-visibility probe ──────────────────────────────────
# The other half of §6c: before setting `-fvisibility=hidden` for a package, ask
# whether that package's library annotates its exports. Compile its real sources
# under its real flags with the flag added and read visibility off the IR — the
# compiler resolves the export macro, so nothing parses a header. Same cheap-probe
# placement rationale as §6b/§6c.

include(joinpath(TEST_DIR, "test_visibility_probe.jl"))

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

# ── 7e. Tier 1 binds a library's own symbols by alias, not by name ───────────
# The one Tier 1 test that runs here, on John's call (2026-10-24): Hub recipes
# do enable slicing (zlib, lua, sqlite, …), and a slice that declared another
# public function of its library by name bound whichever library defined that
# name first in the process — Julia's own libz, for the Hub's zlib. Small
# fixture named after zlib's exports, a decoy with the same names loaded first,
# every call in a child. Needs the C bucket only.

include(joinpath(TEST_DIR, "test_tier1_alias_binding.jl"))

# ── 7f. [wrap] shim_defines: a header's implementation from the TOML, live ────
# A single-header library built with no source file and no macro entry, then
# beside a library source that includes the same header: the define reaches
# only the shim file, so the implementation is defined once. With the define in
# [compile] flags the link stops. Needs the C bucket only.

include(joinpath(TEST_DIR, "test_shim_defines_live.jl"))

# ── 8. Nested-struct ABI resolution (pure ccall path) ────────────────────────
# Library-free trace: structs with struct-typed members must come out with
# verified named fields (SysV register classes preserved by value), and
# unreproducible layouts (packed floats) must refuse by-value crossings loudly.

include(joinpath(TEST_DIR, "test_abi_nested.jl"))

# ── 8b. Tier 3 by-value record returns, live (C1) ─────────────────────────────
# Where C returns a record (registers or a hidden pointer) is per target, and an
# sret the callee does not expect shifts every argument with no error. Calls
# every function of a library-free fixture and checks each argument and return
# member, then checks the classifier against clang's lowering for both targets.

include(joinpath(TEST_DIR, "test_c_sret_abi.jl"))

# ── 8c. C++ by-value record routing, live ─────────────────────────────────────
# A natural record whose Julia type is the C layout goes to Tier 3; a packed one
# (at any depth) or a ≤16-byte byte blob goes to the thunk. Checks the tier of
# every function and calls each one. Needs clang++ and libJLCS.

include(joinpath(TEST_DIR, "test_cpp_record_routing.jl"))

# ── 8d. Emission determinism on a real C/C++ build ───────────────────────────
# A C unit and two C++ units define the same types differently, two pimpl
# classes share `impl`, and a lambda sits in a std::function. The extracted
# tables must not move with the DIE offsets or the unit order, the closure type
# must not be emitted, and the wrapper must not follow the metadata's key order.
# Needs clang++ and libJLCS.

include(joinpath(TEST_DIR, "test_determinism_fixture.jl"))

# ── 8e. A name several types share never crosses by value, live ──────────────
# Two C++ units give `impl`, `Mode` and `Kind` two definitions each, and two C
# files each define their own `struct state`. Every function one of them
# crosses by value must be a trap with no thunk; the rest must keep their tier
# and return the right values. The C++ half needs clang++ and libJLCS.

include(joinpath(TEST_DIR, "test_conflict_trap_fixture.jl"))

# ── 9. Per-file IR cache invalidation on compile-config change ────────────────
# A compile-flag change must invalidate the per-file IR cache without a manual
# `rm -rf` — otherwise stale IR (built with the old flags) is silently reused.

include(joinpath(TEST_DIR, "test_cache_invalidation.jl"))

# ── 9b. A relink of unchanged IR gives the same library ─────────────────────
# lld writes the link time into a PE image, so every relink of an unchanged
# project gave a different DLL and a different LIBRARY_SHA, the one line a
# rebuilt llamacpp's wrapper differed in. Builds a C and a C++ fixture twice
# from the same per-file IR, a second apart, and links an executable twice: the
# bytes must match, and on PE both link-time fields must be 0.

include(joinpath(TEST_DIR, "test_reproducible_link.jl"))

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

# ── 12b. SysV struct classification against LLVM's ABI library ───────────────
# FunctionGen's DWARF-fact classification, as buildCallShape acts on it with
# the SysV target on any host, diffed against LLVM 23's own x86-64 classifier
# (llvm/ABI) over abi_fuzz's records and hand-written ones. Every disagreement
# fails; the report names the shape of each. Skips unless libJLCS was built
# with the oracle (LLVM 23+).

include(joinpath(TEST_DIR, "test_sysv_abi_oracle.jl"))

# ── 12c. SysV record classes, live ───────────────────────────────────────────
# One record per way a by-value record is classified under SysV, built as C++
# at -O0 and -O2, every take_R/make_R called through its Tier 2 thunk with the
# SSE registers poisoned, values checked against a C-side image. The families
# the oracle used to list as @test_broken all returned garbage live before the
# classification moved from the emitted body to the DWARF facts (2026-10-24).

include(joinpath(TEST_DIR, "test_sysv_live.jl"))

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

# ── 19. A link with more inputs than one Windows command line holds ──────────
# llamacpp with ggml-vulkan (384 TUs) could not link from the Hub's checkout
# depth: ~39,000 characters of llvm-link command line, past the 32,767
# CreateProcess accepts. Long input lists now go to a response file, which
# llvm-link reads by Windows rules and clang by GNU rules; both real tools read
# one here, on every host. Then 200 one-line TUs at a depth that passes the limit
# on Windows build end to end. The writer itself is in test_response_files.jl
# (runtests).

include(joinpath(TEST_DIR, "test_link_cmdline_limit.jl"))
