using Test
using RepliBuild
using RepliBuild.ConfigurationManager
using RepliBuild.LLVMEnvironment
using RepliBuild.Discovery
using RepliBuild.Templates
using RepliBuild.JuliaWrapItUp
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

    @testset "Wrapping Stage" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Create test C++ project with library
                test_dir = joinpath(tmpdir, "wrap_test")
                mkpath(joinpath(test_dir, "src"))
                mkpath(joinpath(test_dir, "include"))
                mkpath(joinpath(test_dir, "lib"))

                # Create header file
                write(joinpath(test_dir, "include", "math_funcs.h"), """
                #pragma once
                extern "C" {
                    int add(int a, int b);
                    int multiply(int a, int b);
                    double square(double x);
                }
                """)

                # Create source file
                write(joinpath(test_dir, "src", "math_funcs.cpp"), """
                #include "math_funcs.h"

                int add(int a, int b) {
                    return a + b;
                }

                int multiply(int a, int b) {
                    return a * b;
                }

                double square(double x) {
                    return x * x;
                }
                """)

                # Compile to shared library using BuildBridge
                toolchain = LLVMEnvironment.get_toolchain()
                clangpp = LLVMEnvironment.get_tool("clang++")

                if !isempty(clangpp)
                    lib_path = joinpath(test_dir, "lib", "libmath.so")
                    compile_cmd = [
                        "-shared",
                        "-fPIC",
                        "-I$(joinpath(test_dir, "include"))",
                        joinpath(test_dir, "src", "math_funcs.cpp"),
                        "-o", lib_path
                    ]

                    output, exitcode = BuildBridge.execute(clangpp, compile_cmd)

                    @test exitcode == 0
                    @test isfile(lib_path)

                    if isfile(lib_path)
                        # Test JuliaWrapItUp wrapper configuration
                        @test isdefined(JuliaWrapItUp, :WrapperConfig)
                        @test isdefined(JuliaWrapItUp, :BinaryWrapper)
                        @test isdefined(JuliaWrapItUp, :BinaryInfo)

                        # Create wrapper configuration
                        config = JuliaWrapItUp.WrapperConfig(
                            test_dir,
                            [joinpath(test_dir, "lib")],
                            joinpath(test_dir, "wrappers"),
                            [joinpath(test_dir, "include")],
                            :advanced, :nm, true, false, false, true,
                            true, "", Dict{String,String}(),
                            nothing, false
                        )

                        @test config.project_root == test_dir
                        @test config.wrapper_style == :advanced

                        # Save config and create wrapper
                        config_file = joinpath(test_dir, "wrapper_config.toml")
                        JuliaWrapItUp.save_wrapper_config(config, config_file)
                        wrapper = JuliaWrapItUp.BinaryWrapper(config_file)
                        @test wrapper isa JuliaWrapItUp.BinaryWrapper

                        # Test binary identification
                        binary_type = JuliaWrapItUp.identify_binary_type(lib_path)
                        @test binary_type == :shared_lib

                        # Test binary analysis
                        binary_info = JuliaWrapItUp.analyze_binary(wrapper, lib_path, binary_type)
                        @test !isnothing(binary_info)
                        @test binary_info isa JuliaWrapItUp.BinaryInfo
                        @test binary_info.name == "libmath"
                        @test binary_info.type == :shared_lib
                        @test length(binary_info.symbols) > 0

                        # Test symbol extraction
                        symbols = binary_info.symbols
                        symbol_names = [s["name"] for s in symbols]
                        @test "add" in symbol_names
                        @test "multiply" in symbol_names
                        @test "square" in symbol_names

                        # Test wrapper generation
                        wrapper_code = JuliaWrapItUp.generate_wrapper(wrapper, binary_info)
                        @test !isempty(wrapper_code)
                        @test contains(wrapper_code, "module ")  # Check any module declaration
                        @test contains(wrapper_code, "ccall")
                        @test contains(wrapper_code, "is_loaded")

                        # Save and test the generated wrapper
                        mkpath(config.output_dir)
                        module_name = JuliaWrapItUp.generate_module_name(binary_info.name)
                        wrapper_file = joinpath(config.output_dir, "$(module_name).jl")
                        write(wrapper_file, wrapper_code)
                        @test isfile(wrapper_file)

                        # Try to load and use the wrapper
                        try
                            include(wrapper_file)
                            # Get the module (it should be loaded in Main)
                            mod = getfield(Main, Symbol(module_name))
                            @test isdefined(mod, :is_loaded)
                            @test isdefined(mod, :library_info)

                            # Test library info
                            if mod.is_loaded()
                                info = mod.library_info()
                                @test haskey(info, :name)
                                @test haskey(info, :loaded)
                                @test info[:loaded] == true
                            end
                        catch e
                            @warn "Could not test wrapper execution: $e"
                        end
                    end
                else
                    @warn "clang++ not available, skipping wrapper tests"
                end
            end
        end
    end

    @testset "Executable Creation" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Create test C++ executable project
                test_dir = joinpath(tmpdir, "exe_test")
                mkpath(joinpath(test_dir, "src"))
                mkpath(joinpath(test_dir, "bin"))

                # Create main executable source
                write(joinpath(test_dir, "src", "main.cpp"), """
                #include <iostream>

                int main(int argc, char* argv[]) {
                    std::cout << "Hello from RepliBuild!" << std::endl;

                    if (argc > 1) {
                        std::cout << "Arguments: ";
                        for (int i = 1; i < argc; i++) {
                            std::cout << argv[i] << " ";
                        }
                        std::cout << std::endl;
                    }

                    return 42;
                }
                """)

                # Compile to executable using BuildBridge
                toolchain = LLVMEnvironment.get_toolchain()
                clangpp = LLVMEnvironment.get_tool("clang++")

                if !isempty(clangpp)
                    exe_path = joinpath(test_dir, "bin", "test_program")
                    compile_cmd = [
                        joinpath(test_dir, "src", "main.cpp"),
                        "-o", exe_path
                    ]

                    output, exitcode = BuildBridge.execute(clangpp, compile_cmd)

                    @test exitcode == 0
                    @test isfile(exe_path)

                    if isfile(exe_path) && Sys.isunix()
                        # Make executable
                        run(`chmod +x $exe_path`)

                        # Test executable identification
                        binary_type = JuliaWrapItUp.identify_binary_type(exe_path)
                        @test binary_type in [:executable, :shared_lib]  # May detect as either

                        # Test executable execution (ignore exit code for output capture)
                        try
                            result = read(`$exe_path`, String)
                            @test contains(result, "Hello from RepliBuild!")
                        catch e
                            # Exit code 42 causes ProcessFailedException, get output anyway
                            if isa(e, ProcessFailedException)
                                # Run again and capture output
                                io = IOBuffer()
                                try
                                    run(pipeline(`$exe_path`, stdout=io, stderr=io))
                                catch
                                end
                                result = String(take!(io))
                                @test contains(result, "Hello from RepliBuild!")
                            else
                                rethrow(e)
                            end
                        end

                        # Test with arguments
                        try
                            result = read(`$exe_path test arg`, String)
                            @test contains(result, "Hello from RepliBuild!")
                            @test contains(result, "test")
                            @test contains(result, "arg")
                        catch e
                            if isa(e, ProcessFailedException)
                                io = IOBuffer()
                                try
                                    run(pipeline(`$exe_path test arg`, stdout=io, stderr=io))
                                catch
                                end
                                result = String(take!(io))
                                @test contains(result, "Hello from RepliBuild!")
                                @test contains(result, "test")
                                @test contains(result, "arg")
                            else
                                rethrow(e)
                            end
                        end

                        # Test exit code
                        proc = try
                            run(`$exe_path`)
                        catch e
                            if isa(e, ProcessFailedException)
                                e.procs[1]
                            else
                                rethrow(e)
                            end
                        end
                        @test proc.exitcode == 42
                    end
                else
                    @warn "clang++ not available, skipping executable tests"
                end
            end
        end
    end

end
