#!/usr/bin/env julia
# Precompilation statements for RepliBuild
# This file helps PackageCompiler.jl create an optimized sysimage

using RepliBuild

println("Running RepliBuild precompilation...")

# Test the main user-facing API functions
try
    # Core information functions
    RepliBuild.info()
    RepliBuild.help()

    # LLVM toolchain functions (safe to call)
    toolchain = RepliBuild.get_toolchain()
    RepliBuild.verify_toolchain()
    RepliBuild.print_toolchain_info()

    # Create temporary test directories for initialization
    mktempdir() do tmpdir
        # Test C++ project initialization
        cpp_dir = joinpath(tmpdir, "test_cpp")
        RepliBuild.init(cpp_dir)

        # Test binary project initialization
        bin_dir = joinpath(tmpdir, "test_binary")
        RepliBuild.init(bin_dir, type=:binary)

        # Test available templates
        RepliBuild.available_templates()
    end

    println("✅ Precompilation statements executed successfully")
catch e
    @warn "Some precompilation statements failed (this is OK)" exception=e
    println("✅ Precompilation completed with warnings")
end
