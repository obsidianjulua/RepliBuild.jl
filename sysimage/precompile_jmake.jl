#!/usr/bin/env julia
# Precompilation statements for RepliBuild
# This file helps PackageCompiler.jl create an optimized sysimage

println("Running RepliBuild precompilation...")

# Import core modules to force compilation
using RepliBuild
using RepliBuild.ConfigurationManager
using RepliBuild.LLVMEnvironment
using RepliBuild.Discovery
using RepliBuild.BuildBridge
using RepliBuild.UXHelpers
using RepliBuild.ErrorLearning
using RepliBuild.Templates

# Test core API functions
try
    println("  • Testing LLVMEnvironment...")
    # Initialize LLVM toolchain (critical for all operations)
    toolchain = LLVMEnvironment.get_toolchain()
    println("    ✓ LLVM toolchain initialized: $(toolchain.source)")

    # Verify toolchain is functional
    LLVMEnvironment.verify_toolchain()
    println("    ✓ LLVM toolchain verified")

    println("  • Testing ConfigurationManager...")
    # Test configuration loading/creation
    mktempdir() do tmpdir
        config_path = joinpath(tmpdir, "replibuild.toml")
        config = ConfigurationManager.create_default_config(config_path)
        println("    ✓ Default config created")

        # Test config accessors
        ConfigurationManager.get_include_dirs(config)
        ConfigurationManager.get_source_files(config)
        ConfigurationManager.is_stage_enabled(config, :compile)
        println("    ✓ Config accessors tested")
    end

    println("  • Testing Templates...")
    # Test template system
    Templates.list_templates()
    println("    ✓ Templates loaded")

    println("  • Testing BuildBridge...")
    # Test tool discovery
    BuildBridge.discover_llvm_tools()
    println("    ✓ LLVM tools discovered")

    println("  • Testing ErrorLearning...")
    # Test error learning initialization
    mktempdir() do tmpdir
        db_path = joinpath(tmpdir, "test_errors.db")
        db = ErrorLearning.init_db(db_path)
        println("    ✓ Error learning DB initialized")
    end

    println("\n✅ All precompilation statements executed successfully")
    println("📊 Precompiled modules:")
    println("   • RepliBuild core")
    println("   • ConfigurationManager")
    println("   • LLVMEnvironment")
    println("   • Discovery")
    println("   • BuildBridge")
    println("   • ErrorLearning")
    println("   • Templates")
    println("   • UXHelpers")

catch e
    @warn "Some precompilation statements failed" exception=(e, catch_backtrace())
    println("⚠️  Precompilation completed with warnings (this may be OK)")
end
