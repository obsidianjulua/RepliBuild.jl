#!/usr/bin/env julia
# test_wrapper_simple.jl
# Simplest possible test: Load metadata JSON and generate wrapper

using RepliBuild
using JSON

println("="^70)
println("SIMPLE WRAPPER TEST - Direct metadata → Julia bindings")
println("="^70)
println()

test_dir = @__DIR__
test_cpp = joinpath(test_dir, "test_advanced_types.cpp")
output_dir = joinpath(test_dir, "build_simple")
mkpath(output_dir)

# Step 1: Compile
println("Step 1: Compiling C++ with -g...")
lib_path = joinpath(output_dir, "libtest.so")
run(`clang++ -shared -fPIC -g -O0 -std=c++17 $test_cpp -o $lib_path`)
println("   ✓ Compiled: $lib_path")
println()

# Step 2: Extract metadata
println("Step 2: Extracting metadata...")
using RepliBuild: Compiler

(return_types, struct_defs) = Compiler.extract_dwarf_return_types(lib_path)
symbols = Compiler.extract_symbols_from_binary(lib_path)

# Build functions list for metadata
functions = []
for (symbol, return_info) in return_types
    func = Dict{String,Any}(
        "name" => symbol,
        "mangled" => symbol,
        "demangled" => symbol,
        "return_type" => return_info,
        "parameters" => []  # Simplified - no parameters for this test
    )
    push!(functions, func)
end

metadata = Dict{String,Any}(
    "functions" => functions,
    "struct_definitions" => struct_defs,
    "return_types" => return_types,
    "symbols" => symbols
)

println("   ✓ Extracted:")
println("      - Functions: $(length(functions))")
println("      - Structs: $(count(k -> !startswith(k, "__enum__"), keys(struct_defs)))")
println("      - Enums: $(count(k -> startswith(k, "__enum__"), keys(struct_defs)))")
println()

# Step 3: Save metadata
metadata_file = joinpath(output_dir, "compilation_metadata.json")
open(metadata_file, "w") do f
    JSON.print(f, metadata, 2)
end
println("   ✓ Saved metadata: $metadata_file")
println()

# Step 4: Generate wrapper using RepliBuild API
println("Step 3: Generating wrapper with RepliBuild.wrap()...")

# Create config TOML
config_content = """
[project]
name = "TestAdvanced"
root = "$(test_dir)"

[paths]
output = "$(output_dir)"

[binary]
type = "shared"

[wrap]
enabled = true
style = "clang"
"""

config_file = joinpath(output_dir, "replibuild.toml")
write(config_file, config_content)

# Load config
using RepliBuild: ConfigurationManager
config = ConfigurationManager.load_config(config_file)

# Now call wrap_introspective with proper config
using RepliBuild: Wrapper

# The wrap function expects metadata.json in same dir as library
# It was already saved above

# Call the main wrapper function
wrapper_file = Wrapper.wrap_introspective(config, lib_path, String[])

println("   ✓ Generated: $wrapper_file")
println()

# Step 5: Show the generated code
println("="^70)
println("GENERATED WRAPPER CONTENT")
println("="^70)
println()

wrapper_content = read(wrapper_file, String)

# Show enum section
if contains(wrapper_content, "# Enum Definitions")
    println("📋 ENUM SECTION:")
    enum_lines = split(wrapper_content, '\n')
    in_enum = false
    count = 0
    for line in enum_lines
        if contains(line, "# Enum Definitions")
            in_enum = true
        elseif in_enum && (contains(line, "# Struct Definitions") || contains(line, "export"))
            break
        elseif in_enum
            println(line)
            count += 1
            if count > 40  # Limit output
                println("   ... (truncated)")
                break
            end
        end
    end
    println()
end

# Show struct section
if contains(wrapper_content, "# Struct Definitions")
    println("📦 STRUCT SECTION:")
    struct_lines = split(wrapper_content, '\n')
    in_struct = false
    count = 0
    for line in struct_lines
        if contains(line, "# Struct Definitions")
            in_struct = true
        elseif in_struct && contains(line, "export")
            break
        elseif in_struct
            println(line)
            count += 1
            if count > 50
                println("   ... (truncated)")
                break
            end
        end
    end
    println()
end

println("="^70)
println("VALIDATION")
println("="^70)
println()

# Try to load the generated module
println("Loading generated module...")
try
    include(wrapper_file)
    TestAdvanced = Main.TestAdvanced
    println("   ✓ Module loaded successfully!")

    # Test enum access
    println()
    println("Testing enum access:")
    if isdefined(TestAdvanced, :Color)
        Color = TestAdvanced.Color
        println("   ✓ Color enum exists")
        println("      Red = $(Int(Color.Red))")
        println("      Green = $(Int(Color.Green))")
        println("      Blue = $(Int(Color.Blue))")
    else
        println("   ✗ Color enum not found")
    end

    if isdefined(TestAdvanced, :Status)
        Status = TestAdvanced.Status
        println("   ✓ Status enum exists")
        println("      Idle = $(Int(Status.Idle))")
        println("      Running = $(Int(Status.Running))")
    else
        println("   ✗ Status enum not found")
    end

    # Test struct access
    println()
    println("Testing struct access:")
    if isdefined(TestAdvanced, :Matrix3x3)
        Matrix3x3 = TestAdvanced.Matrix3x3
        println("   ✓ Matrix3x3 exists")
        println("      Fields: $(fieldnames(Matrix3x3))")
        println("      Field type: $(fieldtype(Matrix3x3, :data))")
    else
        println("   ✗ Matrix3x3 not found")
    end

catch e
    println("   ✗ Error loading module:")
    showerror(stdout, e, catch_backtrace())
    println()
end

println()
println("="^70)
println("TEST COMPLETE")
println("="^70)
println("Full wrapper file: $wrapper_file")
println()
