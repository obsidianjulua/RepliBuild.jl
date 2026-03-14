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
    ("rust_test",          "Rust FFI (cdylib, repr(C))"),
]

for (name, label) in INTEGRATION_TESTS
    @testset "$label" begin
        dir = joinpath(TEST_DIR, name)
        @test isdir(dir)

        clean_test_dir(dir)
        toml = joinpath(dir, "replibuild.toml")
        if !isfile(toml)
            toml = RepliBuild.discover(dir, force=true, build=true, wrap=true)
        else
            RepliBuild.build(toml)
            RepliBuild.wrap(toml)
        end
        @test isfile(toml)
        @test isdir(joinpath(dir, "julia"))

        @test run_verify(dir; label=label)
    end
end

# ── 3. Registry tests ────────────────────────────────────────────────────────

include(joinpath(TEST_DIR, "test_registry.jl"))
