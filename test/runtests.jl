#!/usr/bin/env julia
# RepliBuild.jl Test Suite
# Tests the full RepliBuild workflow using the stress_test C++ project

using Test
using RepliBuild

@testset "RepliBuild.jl Tests" begin
    # Get the stress test directory
    test_dir = joinpath(@__DIR__, "stress_test")
    @test isdir(test_dir)

    # Verify the stress test has required files
    @testset "Stress Test Files" begin
        @test isfile(joinpath(test_dir, "include", "numerics.h"))
        @test isfile(joinpath(test_dir, "src", "numerics.cpp"))
    end

    # Test discovery
    @testset "Discovery" begin
        println("\n" * "="^70)
        println("Testing RepliBuild.discover()")
        println("="^70)

        # Clean any existing artifacts
        for dir in ["build", "julia", ".replibuild_cache", "replibuild.toml"]
            path = joinpath(test_dir, dir)
            if ispath(path)
                rm(path, recursive=true, force=true)
            end
        end

        # Run discovery (can use either RepliBuild.discover or RepliBuild.Discovery.discover)
        toml_path = RepliBuild.discover(test_dir, force=true)
        @test isfile(toml_path)
        @test endswith(toml_path, "replibuild.toml")
        @test dirname(toml_path) == abspath(test_dir)
    end

    # Test building
    @testset "Build" begin
        println("\n" * "="^70)
        println("Testing RepliBuild.build()")
        println("="^70)

        toml_path = joinpath(test_dir, "replibuild.toml")
        @test isfile(toml_path)

        # Build the project
        library_path = RepliBuild.build(toml_path)
        @test isfile(library_path)

        # Verify build outputs
        julia_dir = joinpath(test_dir, "julia")
        @test isdir(julia_dir)

        # Check for library file
        lib_files = filter(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"),
                          readdir(julia_dir))
        @test length(lib_files) > 0

        # Check for metadata
        metadata_path = joinpath(julia_dir, "compilation_metadata.json")
        @test isfile(metadata_path)
    end

    # Test wrapping
    @testset "Wrap" begin
        println("\n" * "="^70)
        println("Testing RepliBuild.wrap()")
        println("="^70)

        toml_path = joinpath(test_dir, "replibuild.toml")
        @test isfile(toml_path)

        # Generate wrapper
        wrapper_path = RepliBuild.wrap(toml_path)
        @test isfile(wrapper_path)
        @test endswith(wrapper_path, ".jl")

        # Try to load the wrapper
        println("\n" * "="^70)
        println("Testing generated wrapper loading")
        println("="^70)

        include(wrapper_path)
        # The module name is based on the project name (stress_test → StressTest)
        @test isdefined(Main, :StressTest)
    end

    # Test basic functionality of wrapped library
    @testset "Wrapped Library Functionality" begin
        println("\n" * "="^70)
        println("Testing wrapped library functions")
        println("="^70)

        # Verify the module was loaded and has functions
        @test isdefined(Main, :StressTest)

        # Check that the module has some expected functions
        if isdefined(Main, :StressTest)
            mod = Main.StressTest
            # These functions should exist based on the stress test
            @test isdefined(mod, :vector_dot) || isdefined(mod, :dense_matrix_create)
            println("✓ Wrapper successfully generated and loadable with functions")
        end
    end

    # Test info command
    @testset "Info" begin
        println("\n" * "="^70)
        println("Testing RepliBuild.info()")
        println("="^70)

        toml_path = joinpath(test_dir, "replibuild.toml")
        @test isfile(toml_path)

        # This should not throw
        @test_nowarn RepliBuild.info(toml_path)
    end

    # Test clean command
    @testset "Clean" begin
        println("\n" * "="^70)
        println("Testing RepliBuild.clean()")
        println("="^70)

        toml_path = joinpath(test_dir, "replibuild.toml")
        @test isfile(toml_path)

        # Clean should not throw
        @test_nowarn RepliBuild.clean(toml_path)

        # Verify artifacts are removed
        @test !isdir(joinpath(test_dir, "build"))
        @test !isdir(joinpath(test_dir, "julia"))
        @test !isdir(joinpath(test_dir, ".replibuild_cache"))
    end

    # Test chained workflow
    @testset "Chained Workflow" begin
        println("\n" * "="^70)
        println("Testing chained discover → build → wrap")
        println("="^70)

        # Clean first
        for dir in ["build", "julia", ".replibuild_cache", "replibuild.toml"]
            path = joinpath(test_dir, dir)
            if ispath(path)
                rm(path, recursive=true, force=true)
            end
        end

        # Run full pipeline
        toml_path = RepliBuild.discover(test_dir, force=true, build=true, wrap=true)

        @test isfile(toml_path)

        # Verify all outputs exist
        julia_dir = joinpath(test_dir, "julia")
        @test isdir(julia_dir)

        lib_files = filter(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"),
                          readdir(julia_dir))
        @test length(lib_files) > 0

        jl_files = filter(f -> endswith(f, ".jl"), readdir(julia_dir))
        @test length(jl_files) > 0
    end
end

println("\n" * "="^70)
println("All RepliBuild tests completed!")
println("="^70)

include("test_mlir.jl")
include("test_stdlib_mlir.jl")
