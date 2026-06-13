#!/usr/bin/env julia
# RepliBuild.jl — Full Integration Test Suite
#
# Runs the complete pipeline for each test project: discover → build → wrap → verify.
# Requires: LLVM 21+, Clang, CMake on the system.
# Usage:  julia --project=. test/devtests.jl

using Test
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
    ("stl_test",           "STL Templates (vector, string, map)"),
    ("c_test",             "C Fundamentals (structs, enums, LTO, packed, unions)"),
    ("c_abomination_test", "C Edge Cases (opaque structs, nested callbacks)"),
    ("callback_test",      "Callbacks (Julia ↔ C++)"),
]

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
        # tomls are hand-rolled and must NOT be discovered.)
        toml = RepliBuild.discover(dir, force=true, build=true, wrap=true)
        @test isfile(toml)
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

# ── 8. Nested-struct ABI resolution (pure ccall path) ────────────────────────
# Library-free trace: structs with struct-typed members must come out with
# verified named fields (SysV register classes preserved by value), and
# unreproducible layouts (packed floats) must refuse by-value crossings loudly.

include(joinpath(TEST_DIR, "test_abi_nested.jl"))
