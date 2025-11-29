#!/usr/bin/env julia
# Build test library and generate wrappers - validate binding quality

using RepliBuild

println("="^70)
println("BUILD & WRAP TEST - Quality Validation")
println("="^70)

# Setup paths
test_dir = @__DIR__
cpp_file = joinpath(test_dir, "dwarf_test.cpp")
toml_file = joinpath(test_dir, "replibuild_test.toml")

# Create minimal config
println("\n▶ Step 1: Creating RepliBuild config")

config_content = """
[project]
name = "DwarfTest"
root = "$test_dir"

[paths]
source = "."
output = "julia_bindings"
build = "build"

[compile]
source_files = ["dwarf_test.cpp"]
flags = ["-std=c++17", "-fPIC", "-g", "-O0"]

[binary]
type = "shared"

[wrap]
enabled = true
style = "clang"
module_name = "DwarfTestBindings"
"""

write(toml_file, config_content)
println("✓ Config created: $toml_file")

# Load config and build
println("\n▶ Step 2: Loading config")
config = RepliBuild.ConfigurationManager.load_config(toml_file)

println("\n▶ Step 3: Compiling C++ to library")
RepliBuild.build(toml_file)

println("\n▶ Step 4: Generating Julia wrappers")
RepliBuild.wrap(toml_file)

println("\n" * "="^70)
println("✅ BUILD COMPLETE")
println("="^70)

# Show what was generated
bindings_dir = joinpath(test_dir, "julia_bindings")
if isdir(bindings_dir)
    println("\n📁 Generated files:")
    for f in readdir(bindings_dir, join=true)
        size = filesize(f) / 1024
        println("   $(basename(f))  ($(round(size, digits=1)) KB)")
    end

    # Find the wrapper file
    wrapper_files = filter(f -> endswith(f, ".jl"), readdir(bindings_dir, join=true))
    if !isempty(wrapper_files)
        wrapper = wrapper_files[1]
        println("\n📄 Wrapper module: $(basename(wrapper))")

        # Count what was wrapped
        content = read(wrapper, String)
        function_count = count("function ", content)
        struct_count = count("mutable struct ", content)
        enum_count = count("@enum ", content)

        println("   Functions: $function_count")
        println("   Structs:   $struct_count")
        println("   Enums:     $enum_count")
    end
end

println("\n▶ Next: Load the wrapper and test it!")
println("   julia> include(\"julia_bindings/DwarfTestBindings.jl\")")
println("   julia> using .DwarfTestBindings")
