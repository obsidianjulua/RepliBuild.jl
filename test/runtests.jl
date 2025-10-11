using Test
using RepliBuild
using Logging

# Suppress informational output during tests
test_logger = ConsoleLogger(stderr, Logging.Warn)

@testset "RepliBuild.jl" begin

    @testset "Package Loading" begin
        @test isdefined(RepliBuild, :VERSION)
        @test RepliBuild.VERSION isa VersionNumber
        @test RepliBuild.VERSION >= v"0.1.0"
    end

    @testset "Submodules Available" begin
        @test isdefined(RepliBuild, :LLVMEnvironment)
        @test isdefined(RepliBuild, :ConfigurationManager)
        @test isdefined(RepliBuild, :Templates)
        @test isdefined(RepliBuild, :ASTWalker)
        @test isdefined(RepliBuild, :BuildHelpers)
        @test isdefined(RepliBuild, :Discovery)
        @test isdefined(RepliBuild, :ProjectWizard)
        @test isdefined(RepliBuild, :ErrorLearning)
        @test isdefined(RepliBuild, :BuildBridge)
        @test isdefined(RepliBuild, :CMakeParser)
        @test isdefined(RepliBuild, :LLVMake)
        @test isdefined(RepliBuild, :JuliaWrapItUp)
        @test isdefined(RepliBuild, :ClangJLBridge)
        @test isdefined(RepliBuild, :DaemonManager)
    end

    @testset "Exported Functions" begin
        # Core workflow functions
        @test isdefined(RepliBuild, :init)
        @test isdefined(RepliBuild, :compile)
        @test isdefined(RepliBuild, :discover)
        @test isdefined(RepliBuild, :compile_project)

        # Template functions
        @test isdefined(RepliBuild, :create_project_interactive)
        @test isdefined(RepliBuild, :available_templates)
        @test isdefined(RepliBuild, :use_template)

        # Binary wrapping functions
        @test isdefined(RepliBuild, :wrap)
        @test isdefined(RepliBuild, :wrap_binary)
        @test isdefined(RepliBuild, :generate_wrappers)
        @test isdefined(RepliBuild, :scan_binaries)

        # CMake import
        @test isdefined(RepliBuild, :import_cmake)

        # Binding generation
        @test isdefined(RepliBuild, :generate_bindings_clangjl)

        # Info functions
        @test isdefined(RepliBuild, :info)
        @test isdefined(RepliBuild, :help)
        @test isdefined(RepliBuild, :scan)
        @test isdefined(RepliBuild, :analyze)

        # Daemon management
        @test isdefined(RepliBuild, :start_daemons)
        @test isdefined(RepliBuild, :stop_daemons)
        @test isdefined(RepliBuild, :daemon_status)
        @test isdefined(RepliBuild, :ensure_daemons)

        # LLVM toolchain
        @test isdefined(RepliBuild, :get_toolchain)
        @test isdefined(RepliBuild, :verify_toolchain)
        @test isdefined(RepliBuild, :print_toolchain_info)
        @test isdefined(RepliBuild, :with_llvm_env)

        # Advanced functions
        @test isdefined(RepliBuild, :discover_tools)
    end

    @testset "Exported Types" begin
        # LLVMake types
        @test isdefined(RepliBuild, :LLVMJuliaCompiler)
        @test isdefined(RepliBuild, :CompilerConfig)
        @test isdefined(RepliBuild, :TargetConfig)

        # JuliaWrapItUp types
        @test isdefined(RepliBuild, :BinaryWrapper)
        @test isdefined(RepliBuild, :WrapperConfig)
        @test isdefined(RepliBuild, :BinaryInfo)
    end

    @testset "Help and Info Functions" begin
        # Test that help/info functions run without errors
        with_logger(test_logger) do
            @test_nowarn RepliBuild.info()
            @test_nowarn RepliBuild.help()
        end
    end

    @testset "Temporary Project Initialization" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Test C++ project initialization
                cpp_dir = joinpath(tmpdir, "test_cpp_project")
                @test_nowarn RepliBuild.init(cpp_dir)
                @test isdir(cpp_dir)
                @test isdir(joinpath(cpp_dir, "src"))
                @test isdir(joinpath(cpp_dir, "include"))
                @test isdir(joinpath(cpp_dir, "julia"))
                @test isdir(joinpath(cpp_dir, "build"))
                @test isdir(joinpath(cpp_dir, "test"))
                @test isfile(joinpath(cpp_dir, "replibuild.toml"))

                # Test binary project initialization
                bin_dir = joinpath(tmpdir, "test_binary_project")
                @test_nowarn RepliBuild.init(bin_dir, type=:binary)
                @test isdir(bin_dir)
                @test isdir(joinpath(bin_dir, "lib"))
                @test isdir(joinpath(bin_dir, "bin"))
                @test isdir(joinpath(bin_dir, "julia_wrappers"))
                @test isfile(joinpath(bin_dir, "wrapper_config.toml"))
            end
        end
    end

    @testset "LLVM Toolchain Functions" begin
        with_logger(test_logger) do
            # Test that toolchain functions can be called
            @test_nowarn toolchain = RepliBuild.get_toolchain()
            @test_nowarn RepliBuild.verify_toolchain()
            @test_nowarn RepliBuild.print_toolchain_info()
        end
    end

    @testset "Configuration Manager" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Initialize a test project
                test_project = joinpath(tmpdir, "config_test")
                RepliBuild.init(test_project)

                config_file = joinpath(test_project, "replibuild.toml")
                @test isfile(config_file)

                # Test that config file contains expected sections
                config_content = read(config_file, String)
                @test occursin("[paths]", config_content)
            end
        end
    end

    @testset "Error Handling" begin
        # Test that invalid project types are caught
        mktempdir() do tmpdir
            with_logger(test_logger) do
                invalid_dir = joinpath(tmpdir, "invalid_test")
                @test_throws ErrorException RepliBuild.init(invalid_dir, type=:invalid_type)
            end
        end
    end

    @testset "Discovery Module" begin
        mktempdir() do tmpdir
            with_logger(test_logger) do
                # Create a simple test structure
                test_dir = joinpath(tmpdir, "discovery_test")
                mkpath(joinpath(test_dir, "src"))

                # Create a dummy C++ file
                write(joinpath(test_dir, "src", "test.cpp"), """
                #include <iostream>

                int main() {
                    std::cout << "Hello" << std::endl;
                    return 0;
                }
                """)

                # Test scan function
                @test_nowarn result = RepliBuild.scan(test_dir, generate_config=false)
            end
        end
    end

    @testset "Module Functionality" begin
        # Test that submodules are properly loaded and usable
        @test RepliBuild.LLVMEnvironment isa Module
        @test RepliBuild.ConfigurationManager isa Module
        @test RepliBuild.Templates isa Module
        @test RepliBuild.BuildBridge isa Module
        @test RepliBuild.LLVMake isa Module
        @test RepliBuild.JuliaWrapItUp isa Module
    end

end
