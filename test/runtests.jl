using Test
using RepliBuild
using RepliBuild.ConfigurationManager
using RepliBuild.LLVMEnvironment
using RepliBuild.Discovery
using RepliBuild.Templates
using Logging

# Suppress informational output during tests
test_logger = ConsoleLogger(stderr, Logging.Warn)

@testset "RepliBuild.jl" begin

    @testset "Package Loading" begin
        @test isdefined(RepliBuild, :VERSION)
        @test RepliBuild.VERSION isa VersionNumber
        @test RepliBuild.VERSION >= v"0.1.0"
    end

    @testset "Core Modules Available" begin
        # Core architecture modules
        @test isdefined(RepliBuild, :LLVMEnvironment)
        @test isdefined(RepliBuild, :ConfigurationManager)
        @test isdefined(RepliBuild, :Discovery)
        @test isdefined(RepliBuild, :BuildBridge)
        @test isdefined(RepliBuild, :ErrorLearning)
        @test isdefined(RepliBuild, :Templates)
        @test isdefined(RepliBuild, :UXHelpers)

        # Extended modules
        @test isdefined(RepliBuild, :ASTWalker)
        @test isdefined(RepliBuild, :LLVMake)
        @test isdefined(RepliBuild, :JuliaWrapItUp)
        @test isdefined(RepliBuild, :DaemonManager)
    end

    @testset "Core Functions" begin
        # LLVM toolchain functions
        @test isdefined(LLVMEnvironment, :get_toolchain)
        @test isdefined(LLVMEnvironment, :init_toolchain)
        @test isdefined(LLVMEnvironment, :verify_toolchain)
        @test isdefined(LLVMEnvironment, :with_llvm_env)
        @test isdefined(LLVMEnvironment, :get_tool)

        # Configuration management
        @test isdefined(ConfigurationManager, :load_config)
        @test isdefined(ConfigurationManager, :save_config)
        @test isdefined(ConfigurationManager, :create_default_config)
        @test isdefined(ConfigurationManager, :get_include_dirs)
        @test isdefined(ConfigurationManager, :get_source_files)
        @test isdefined(ConfigurationManager, :validate_config)

        # Discovery functions
        @test isdefined(Discovery, :discover)

        # Template functions
        @test isdefined(Templates, :plant)
        @test isdefined(Templates, :initialize_project)
    end

    @testset "Core Types" begin
        # Configuration types
        @test isdefined(ConfigurationManager, :RepliBuildConfig)

        # LLVM types
        @test isdefined(LLVMEnvironment, :LLVMToolchain)
    end

    @testset "Project Initialization" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Test project creation with Templates
                test_project = joinpath(tmpdir, "test_cpp_project")

                # Use Templates.initialize_project with keyword argument
                @test_nowarn Templates.initialize_project(test_project, template=:cpp_project)

                # Verify directory structure was created
                @test isdir(test_project)

                # Test configuration creation
                config_file = joinpath(test_project, "replibuild.toml")
                @test_nowarn config = ConfigurationManager.create_default_config(config_file)
                @test isfile(config_file)
            end
        end
    end

    @testset "LLVM Toolchain" begin
        with_logger(test_logger) do
            # Test toolchain initialization
            @test_nowarn toolchain = LLVMEnvironment.get_toolchain()
            toolchain = LLVMEnvironment.get_toolchain()

            # Verify toolchain properties
            @test toolchain isa LLVMEnvironment.LLVMToolchain
            @test !isempty(toolchain.root)
            @test !isempty(toolchain.bin_dir)
            @test !isempty(toolchain.tools)
            @test toolchain.source in ["intree", "jll", "system", "custom"]

            # Test essential tools exist
            @test LLVMEnvironment.has_tool("clang++")
            @test LLVMEnvironment.has_tool("llvm-config")
            @test LLVMEnvironment.has_tool("llvm-link")
            @test LLVMEnvironment.has_tool("opt")

            # Test verification
            @test LLVMEnvironment.verify_toolchain() == true
        end
    end

    @testset "Configuration System" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Test config creation
                config_file = joinpath(tmpdir, "replibuild.toml")
                config = ConfigurationManager.create_default_config(config_file)

                @test config isa ConfigurationManager.RepliBuildConfig
                @test isfile(config_file)
                @test !isempty(config.project_name)
                @test config.project_root == tmpdir

                # Test config sections exist
                @test haskey(config.discovery, "enabled")
                @test haskey(config.compile, "enabled")
                @test haskey(config.link, "enabled")
                @test haskey(config.binary, "enabled")
                @test haskey(config.wrap, "enabled")

                # Test config accessors
                @test ConfigurationManager.get_include_dirs(config) isa Vector
                @test ConfigurationManager.get_source_files(config) isa Dict
                @test ConfigurationManager.is_stage_enabled(config, :compile) == true

                # Test config validation
                issues = ConfigurationManager.validate_config(config)
                @test issues isa Vector{String}

                # Test config save/load
                ConfigurationManager.save_config(config)
                loaded_config = ConfigurationManager.load_config(config_file)
                @test loaded_config.project_name == config.project_name
            end
        end
    end

    @testset "Discovery System" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Create test C++ project structure
                test_dir = joinpath(tmpdir, "discovery_test")
                mkpath(joinpath(test_dir, "src"))
                mkpath(joinpath(test_dir, "include"))

                # Create dummy C++ files
                write(joinpath(test_dir, "src", "main.cpp"), """
                #include "test.h"
                int main() { return test_function(); }
                """)

                write(joinpath(test_dir, "include", "test.h"), """
                #pragma once
                int test_function();
                """)

                write(joinpath(test_dir, "src", "test.cpp"), """
                #include "test.h"
                int test_function() { return 42; }
                """)

                # Test discovery
                @test_nowarn config = Discovery.discover(test_dir)
                config = Discovery.discover(test_dir)

                # Verify discovery results
                @test isfile(joinpath(test_dir, "replibuild.toml"))
                @test config isa ConfigurationManager.RepliBuildConfig

                # Check discovered files
                files = ConfigurationManager.get_source_files(config)
                @test haskey(files, "cpp_sources")
                @test length(files["cpp_sources"]) >= 2
            end
        end
    end

    @testset "Module Verification" begin
        # Verify all core modules are properly loaded
        @test RepliBuild.LLVMEnvironment isa Module
        @test RepliBuild.ConfigurationManager isa Module
        @test RepliBuild.Discovery isa Module
        @test RepliBuild.BuildBridge isa Module
        @test RepliBuild.ErrorLearning isa Module
        @test RepliBuild.Templates isa Module
        @test RepliBuild.UXHelpers isa Module
    end

end
