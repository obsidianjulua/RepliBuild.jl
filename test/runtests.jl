using Test
using RepliBuild

@testset "RepliBuild.jl" begin
    @testset "API Exports" begin
        # Core API must be exported
        @test isdefined(RepliBuild, :build)
        @test isdefined(RepliBuild, :wrap)
        @test isdefined(RepliBuild, :info)
        @test isdefined(RepliBuild, :clean)

        # Advanced modules must be exported
        @test isdefined(RepliBuild, :Compiler)
        @test isdefined(RepliBuild, :Wrapper)
        @test isdefined(RepliBuild, :Discovery)
        @test isdefined(RepliBuild, :ConfigurationManager)
    end

    @testset "End-to-End: C++ → Julia Binding" begin
        # Create temporary test project
        test_dir = mktempdir()

        try
            # Write test C++ code
            src_dir = joinpath(test_dir, "src")
            mkpath(src_dir)

            cpp_code = """
            // Simple math functions for testing

            struct Point {
                double x;
                double y;
            };

            extern "C" {
                int add(int a, int b) {
                    return a + b;
                }

                double multiply(double a, double b) {
                    return a * b;
                }

                Point create_point(double x, double y) {
                    Point p;
                    p.x = x;
                    p.y = y;
                    return p;
                }
            }
            """

            write(joinpath(src_dir, "test.cpp"), cpp_code)

            # Create replibuild.toml config
            config_toml = """
            [project]
            name = "testlib"
            root = "$test_dir"

            [compile]
            source_files = ["$test_dir/src/test.cpp"]
            include_dirs = []
            flags = ["-std=c++17", "-fPIC", "-g"]

            [binary]
            type = "shared"
            output_name = "libtestlib.so"

            [workflow]
            stages = ["compile", "link", "binary"]

            [llvm]
            toolchain = "auto"

            [cache]
            enabled = false
            """

            write(joinpath(test_dir, "replibuild.toml"), config_toml)

            # Test 1: Build C++ library
            @testset "Build Phase" begin
                original_dir = pwd()
                try
                    cd(test_dir)

                    # Should compile successfully
                    library_path = RepliBuild.build()

                    @test isfile(library_path)
                    @test endswith(library_path, ".so") || endswith(library_path, ".dylib") || endswith(library_path, ".dll")

                    # Should create metadata
                    metadata_path = joinpath(test_dir, "julia", "compilation_metadata.json")
                    @test isfile(metadata_path)

                finally
                    cd(original_dir)
                end
            end

            # Test 2: Generate Julia wrapper
            @testset "Wrap Phase" begin
                original_dir = pwd()
                try
                    cd(test_dir)

                    # Should generate wrapper
                    wrapper_path = RepliBuild.wrap()

                    @test isfile(wrapper_path)
                    @test endswith(wrapper_path, ".jl")

                    # Wrapper should be valid Julia code
                    wrapper_code = read(wrapper_path, String)
                    @test occursin("module", wrapper_code)
                    @test occursin("ccall", wrapper_code)

                finally
                    cd(original_dir)
                end
            end

            # Test 3: Verify generated wrapper exists and is valid Julia
            @testset "Verify Generated Wrapper" begin
                original_dir = pwd()
                try
                    cd(test_dir)

                    # Check wrapper was created
                    wrapper_path = joinpath(test_dir, "julia", "Testlib.jl")
                    @test isfile(wrapper_path)

                    # Verify wrapper contains expected elements
                    wrapper_code = read(wrapper_path, String)
                    @test occursin("module Testlib", wrapper_code)
                    @test occursin("using Libdl", wrapper_code)
                    @test occursin("ccall", wrapper_code) || occursin("dlsym", wrapper_code)

                    # Wrapper should be valid Julia syntax (parse check)
                    @test Meta.parse(wrapper_code) !== nothing

                finally
                    cd(original_dir)
                end
            end

        finally
            # Cleanup
            rm(test_dir, recursive=true, force=true)
        end
    end

    @testset "Configuration Management" begin
        # Test config loading
        test_dir = mktempdir()

        try
            config_content = """
            [project]
            name = "testproject"
            root = "$test_dir"

            [compile]
            source_files = []
            include_dirs = []
            flags = ["-std=c++17"]

            [binary]
            type = "shared"
            """

            config_path = joinpath(test_dir, "replibuild.toml")
            write(config_path, config_content)

            # Should load config without errors
            original_dir = pwd()
            try
                cd(test_dir)
                config = RepliBuild.ConfigurationManager.load_config("replibuild.toml")
                @test config.project.name == "testproject"
            finally
                cd(original_dir)
            end

        finally
            rm(test_dir, recursive=true, force=true)
        end
    end

    @testset "Clean Function" begin
        test_dir = mktempdir()

        try
            # Create fake build artifacts
            build_dir = joinpath(test_dir, "build")
            julia_dir = joinpath(test_dir, "julia")
            cache_dir = joinpath(test_dir, ".replibuild_cache")

            mkpath(build_dir)
            mkpath(julia_dir)
            mkpath(cache_dir)

            write(joinpath(build_dir, "test.ll"), "fake ir")
            write(joinpath(julia_dir, "lib.so"), "fake lib")

            @test isdir(build_dir)
            @test isdir(julia_dir)
            @test isdir(cache_dir)

            # Clean should remove them
            RepliBuild.clean(test_dir)

            @test !isdir(build_dir)
            @test !isdir(julia_dir)
            @test !isdir(cache_dir)

        finally
            rm(test_dir, recursive=true, force=true)
        end
    end
end
