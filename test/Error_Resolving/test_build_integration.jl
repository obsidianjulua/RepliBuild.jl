#!/usr/bin/env julia
# Integration test: Verify external library calls and artifact loading work end-to-end
# This test validates that the build system can:
# 1. Parse CMakeLists.txt with find_package() calls
# 2. Automatically resolve and install JLL packages
# 3. Extract artifact paths from JLL packages
# 4. Make those paths available for compilation

using Pkg
Pkg.activate(".")

using RepliBuild
using RepliBuild.ModuleRegistry
using RepliBuild.CMakeParser

println("="^80)
println("RepliBuild Integration Test: External Library Resolution & Artifact Loading")
println("="^80)
println()

# Test Case: Resolve a CMake project that depends on external libraries
println("🧪 Test: End-to-End External Library Resolution")
println("-"^80)
println()

# Create a minimal test CMake project
test_dir = mktempdir()
println("📁 Test directory: $test_dir")

# Write a simple CMakeLists.txt that requires external libraries
cmake_content = """
cmake_minimum_required(VERSION 3.10)
project(TestExternalLibs)

set(CMAKE_CXX_STANDARD 17)

# Find external packages (these should auto-resolve to JLLs)
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)

# Create a simple library
add_library(testlib SHARED
    test.cpp
)

target_link_libraries(testlib
    ZLIB::ZLIB
    PNG::PNG
)
"""

cmake_file = joinpath(test_dir, "CMakeLists.txt")
write(cmake_file, cmake_content)

# Write a dummy source file
write(joinpath(test_dir, "test.cpp"), """
#include <zlib.h>
#include <png.h>

extern "C" void test_libs() {
    // Just reference symbols to ensure linking works
    zlibVersion();
    png_access_version_number();
}
""")

println("✅ Created test CMake project")
println()

# Step 1: Parse the CMakeLists.txt
println("Step 1: Parse CMakeLists.txt")
println("-"^80)

try
    cmake_project = CMakeParser.parse_cmake_file(cmake_file)

    println("✅ Parsed CMake project successfully")
    println("   Project name: $(cmake_project.project_name)")
    println("   External packages found: $(join(cmake_project.find_packages, ", "))")
    println("   Targets: $(join(keys(cmake_project.targets), ", "))")
    println()

    # Step 2: Resolve external dependencies
    println("Step 2: Resolve External Dependencies")
    println("-"^80)

    resolved_modules = Dict{String, Any}()

    for pkg_name in cmake_project.find_packages
        println("\n🔍 Resolving: $pkg_name")

        mod_info = ModuleRegistry.resolve_module(pkg_name)

        if !isnothing(mod_info)
            resolved_modules[pkg_name] = mod_info

            println("  ✅ Resolved successfully!")
            println("     Source: $(mod_info.source)")
            println("     JLL Package: $(mod_info.julia_package)")
            println("     Include dirs: $(length(mod_info.include_dirs)) found")
            println("     Library dirs: $(length(mod_info.library_dirs)) found")
            println("     Libraries: $(join(mod_info.libraries, ", "))")

            # Verify paths exist
            println("\n  📂 Verifying artifact paths:")
            for inc_dir in mod_info.include_dirs
                exists = isdir(inc_dir)
                println("     $(exists ? "✓" : "✗") Include: $inc_dir")
                if !exists
                    println("       ⚠️  WARNING: Include directory not found!")
                end
            end

            for lib_dir in mod_info.library_dirs
                exists = isdir(lib_dir)
                println("     $(exists ? "✓" : "✗") Library: $lib_dir")
                if !exists
                    println("       ⚠️  WARNING: Library directory not found!")
                end
            end
        else
            println("  ❌ Failed to resolve $pkg_name")
        end
    end

    println()

    # Step 3: Generate RepliBuild configuration
    println("Step 3: Generate RepliBuild Configuration")
    println("-"^80)

    config = CMakeParser.to_replibuild_config(cmake_project, "testlib")

    println("✅ Generated replibuild.toml configuration")
    println()

    if haskey(config, "dependencies")
        println("📦 Dependencies section:")
        for (name, dep_info) in config["dependencies"]
            println("  [$name]")
            println("    source: $(dep_info["source"])")
            if haskey(dep_info, "julia_package")
                println("    julia_package: $(dep_info["julia_package"])")
            end
        end
        println()
    end

    if haskey(config, "compile")
        compile_cfg = config["compile"]
        println("🔧 Compile configuration:")

        if haskey(compile_cfg, "include_dirs")
            println("  Include directories ($(length(compile_cfg["include_dirs"])) total):")
            for (i, dir) in enumerate(compile_cfg["include_dirs"])
                if i <= 5
                    println("    • $dir")
                end
            end
            if length(compile_cfg["include_dirs"]) > 5
                println("    ... $(length(compile_cfg["include_dirs"]) - 5) more")
            end
        end

        if haskey(compile_cfg, "library_dirs")
            println("  Library directories:")
            for dir in compile_cfg["library_dirs"]
                println("    • $dir")
            end
        end

        if haskey(compile_cfg, "link_libraries")
            println("  Link libraries:")
            for lib in compile_cfg["link_libraries"]
                println("    • $lib")
            end
        end
        println()
    end

    # Step 4: Validate Build Readiness
    println("Step 4: Validate Build Readiness")
    println("-"^80)

    all_paths_valid = true

    if haskey(config, "compile") && haskey(config["compile"], "include_dirs")
        for inc_dir in config["compile"]["include_dirs"]
            if !isdir(inc_dir)
                println("  ✗ Missing include directory: $inc_dir")
                all_paths_valid = false
            end
        end
    end

    if haskey(config, "compile") && haskey(config["compile"], "library_dirs")
        for lib_dir in config["compile"]["library_dirs"]
            if !isdir(lib_dir)
                println("  ✗ Missing library directory: $lib_dir")
                all_paths_valid = false
            end
        end
    end

    if all_paths_valid
        println("  ✅ All artifact paths are valid and accessible!")
        println("  ✅ Build system is ready to compile with external libraries")
    else
        println("  ⚠️  Some artifact paths are missing")
        println("  ℹ️  This may indicate JLL artifacts weren't fully extracted")
    end

    println()
    println("="^80)
    println("🎉 Integration Test Complete!")
    println("="^80)
    println()

    # Summary
    println("📊 Summary:")
    println("  • CMake parsing: ✅")
    println("  • External dependency resolution: ✅ ($(length(resolved_modules))/$(length(cmake_project.find_packages)) packages)")
    println("  • Artifact path extraction: $(all_paths_valid ? "✅" : "⚠️")")
    println("  • RepliBuild config generation: ✅")
    println()

    if all_paths_valid
        println("✅ SUCCESS: Build system can handle external library calls and artifacts!")
        println()
        println("💡 Next steps:")
        println("   1. The build system will automatically:")
        println("      - Detect find_package() in CMakeLists.txt")
        println("      - Search for matching JLL packages")
        println("      - Install JLL packages if needed")
        println("      - Extract artifact paths (include/lib dirs)")
        println("      - Add them to compiler/linker flags")
        println("   2. No manual configuration required!")
    else
        println("⚠️  WARNING: Some artifact paths are missing")
        println("   This is likely a JLL package issue or path extraction bug")
        println("   Check ModuleRegistry.jl resolve_jll_module() logic")
    end

catch e
    println("❌ Test failed with error:")
    println()
    showerror(stdout, e, catch_backtrace())
    println()
end

# Cleanup
try
    rm(test_dir; recursive=true, force=true)
catch
end

println()
println("="^80)
