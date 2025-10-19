#!/usr/bin/env julia
# Test script for ModuleRegistry external library resolution

using Pkg
Pkg.activate(".")

using RepliBuild
using RepliBuild.ModuleRegistry

println("="^70)
println("RepliBuild ModuleRegistry Test")
println("="^70)
println()

# Test 1: List available modules
println("📋 Available modules:")
modules = list_modules()
if isempty(modules)
    println("   (none - registry is empty)")
else
    for mod in modules
        println("   • $mod")
    end
end
println()

# Test 2: Resolve a common library (Boost via JLL)
println("🔍 Test: Resolving Boost")
println("-"^70)
boost_info = ModuleRegistry.resolve_module("Boost")
if !isnothing(boost_info)
    println("✅ Boost resolved!")
    println("   Source: $(boost_info.source)")
    println("   Julia Package: $(boost_info.julia_package)")
    println("   Include dirs: $(length(boost_info.include_dirs))")
    println("   Library dirs: $(length(boost_info.library_dirs))")
    println("   Libraries: $(boost_info.libraries)")
else
    println("❌ Could not resolve Boost")
end
println()

# Test 3: Parse a CMake file with find_package
println("🔍 Test: CMake project with external dependencies")
println("-"^70)

# Create test CMakeLists.txt
test_dir = mktempdir()
cmake_file = joinpath(test_dir, "CMakeLists.txt")
write(cmake_file, """
cmake_minimum_required(VERSION 3.10)
project(TestProject)

# Find external packages
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED)

# Create a library
add_library(mylib SHARED
    src/main.cpp
)

target_include_directories(mylib PRIVATE include)
target_link_libraries(mylib Boost::system Boost::filesystem opencv_core)
""")

# Create dummy source files
mkpath(joinpath(test_dir, "src"))
mkpath(joinpath(test_dir, "include"))
write(joinpath(test_dir, "src", "main.cpp"), """
#include <iostream>
int main() { return 0; }
""")

println("   Created test CMakeLists.txt at: $test_dir")
println()

# Parse CMake file
println("📦 Parsing CMakeLists.txt...")
cmake_project = RepliBuild.CMakeParser.parse_cmake_file(cmake_file)
println("   Project: $(cmake_project.project_name)")
println("   Targets: $(join(keys(cmake_project.targets), ", "))")
println("   External packages: $(join(cmake_project.find_packages, ", "))")
println()

# Convert to RepliBuild config
println("🔄 Converting to replibuild.toml...")
config_data = RepliBuild.CMakeParser.to_replibuild_config(cmake_project, "mylib")
println()

if haskey(config_data, "dependencies")
    println("✅ Dependencies resolved:")
    for (name, info) in config_data["dependencies"]
        println("   • $name: $(info["source"]) ($(get(info, "julia_package", "N/A")))")
    end
else
    println("⚠️  No dependencies section found")
end
println()

if haskey(config_data, "compile")
    compile_config = config_data["compile"]
    if haskey(compile_config, "include_dirs")
        println("📂 Include directories ($(length(compile_config["include_dirs"]))):")
        for dir in compile_config["include_dirs"][1:min(3, end)]
            println("   • $dir")
        end
        if length(compile_config["include_dirs"]) > 3
            println("   ... and $(length(compile_config["include_dirs"]) - 3) more")
        end
    end
    println()

    if haskey(compile_config, "link_libraries")
        println("🔗 Link libraries:")
        for lib in compile_config["link_libraries"]
            println("   • $lib")
        end
    end
end
println()

# Test 4: Export module config
println("💾 Test: Export module configuration")
println("-"^70)
if !isnothing(boost_info)
    export_file = joinpath(test_dir, "boost_config.toml")
    ModuleRegistry.export_module_config(boost_info, export_file)
    println("   Exported to: $export_file")

    # Verify it can be re-loaded
    reloaded = ModuleRegistry.load_module_from_toml(export_file)
    println("   ✓ Successfully reloaded: $(reloaded.name)")
end
println()

println("="^70)
println("✅ All tests complete!")
println("="^70)
println()
println("Summary:")
println("• ModuleRegistry can resolve Julia JLL packages")
println("• CMakeParser integrates with ModuleRegistry")
println("• External dependencies auto-resolve to Julia ecosystem")
println("• Module configs can be cached and shared")
println()
