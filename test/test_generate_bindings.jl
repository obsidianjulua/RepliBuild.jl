#!/usr/bin/env julia
# test_generate_bindings.jl
# Full end-to-end test: C++ → compile → DWARF extraction → Julia wrapper generation

using RepliBuild

println("="^70)
println("FULL BINDING GENERATION TEST")
println("Testing: Enums + Arrays + Function Pointers → Julia")
println("="^70)
println()

# Setup test directory
test_dir = @__DIR__
test_cpp = joinpath(test_dir, "test_advanced_types.cpp")
build_dir = joinpath(test_dir, "build_advanced")
output_dir = joinpath(build_dir, "julia")

# Clean previous build
rm(build_dir, force=true, recursive=true)
mkpath(output_dir)

println("Step 1: Compile C++ with debug info...")
test_so = joinpath(output_dir, "libadvanced_types.so")
compile_cmd = `clang++ -shared -fPIC -g -O0 -std=c++17 $test_cpp -o $test_so`
run(compile_cmd)
println("   ✓ Compiled: $test_so")
println()

println("Step 2: Extract compilation metadata...")
using RepliBuild: Compiler

# Extract DWARF and symbols
(return_types, struct_defs) = Compiler.extract_dwarf_return_types(test_so)
symbols = Compiler.extract_symbols_from_binary(test_so)

println("   Extracted:")
println("      - Functions: $(length(return_types))")
println("      - Structs: $(count(k -> !startswith(k, "__enum__"), keys(struct_defs)))")
println("      - Enums: $(count(k -> startswith(k, "__enum__"), keys(struct_defs)))")
println("      - Symbols: $(length(symbols))")
println()

# Build metadata dictionary
metadata = Dict(
    "return_types" => return_types,
    "struct_definitions" => struct_defs,
    "symbols" => symbols,
    "compile_info" => Dict(
        "compiler" => "clang++",
        "flags" => ["-g", "-O0", "-std=c++17"]
    )
)

println("Step 3: Generate Julia wrapper...")
using RepliBuild: Wrapper

# Create minimal config-like object (just need project name for module name)
config = (project = (name = "AdvancedTypes",),)

# Create type registry
registry = Wrapper.create_type_registry(config)

# Generate wrapper using the internal function
wrapper_content = Wrapper.generate_introspective_module(
    config,
    test_so,
    metadata,
    "AdvancedTypes",
    registry,
    true  # generate_docs
)

# Save wrapper
wrapper_file = joinpath(output_dir, "AdvancedTypes.jl")
write(wrapper_file, wrapper_content)

println("   ✓ Generated: $wrapper_file")
println("   ✓ Size: $(length(wrapper_content)) bytes")
println()

println("="^70)
println("GENERATED WRAPPER PREVIEW")
println("="^70)
println()

# Show enum section
enum_section = match(r"# Enum Definitions.*?(?=# Struct Definitions|# =======)"s, wrapper_content)
if !isnothing(enum_section)
    println("ENUMS:")
    println(enum_section.match)
else
    println("⚠ No enums found in wrapper")
end
println()

# Show struct section
struct_section = match(r"# Struct Definitions.*?(?=export|# Function Wrappers)"s, wrapper_content)
if !isnothing(struct_section)
    lines = split(struct_section.match, '\n')
    println("STRUCTS (first 30 lines):")
    for (i, line) in enumerate(lines[1:min(30, length(lines))])
        println(line)
    end
    if length(lines) > 30
        println("   ... ($(length(lines) - 30) more lines)")
    end
else
    println("⚠ No structs found in wrapper")
end
println()

# Count functions
func_count = count(r"^function ", wrapper_content)
println("FUNCTIONS: $func_count generated")
println()

println("="^70)
println("VALIDATION TEST")
println("="^70)
println()

println("Step 4: Load and test generated wrapper...")
try
    include(wrapper_file)
    println("   ✓ Wrapper loaded successfully!")

    # Try to access the module
    AdvancedTypes = eval(Symbol("AdvancedTypes"))

    # Check if enums are accessible
    println()
    println("   Testing enum access:")
    try
        Color = getfield(AdvancedTypes, :Color)
        println("      ✓ Color enum found")
        println("         - Red = $(Int(Color.Red))")
        println("         - Green = $(Int(Color.Green))")
        println("         - Blue = $(Int(Color.Blue))")
    catch e
        println("      ✗ Color enum not accessible: $e")
    end

    try
        Status = getfield(AdvancedTypes, :Status)
        println("      ✓ Status enum found")
        println("         - Idle = $(Int(Status.Idle))")
        println("         - Running = $(Int(Status.Running))")
        println("         - Stopped = $(Int(Status.Stopped))")
    catch e
        println("      ✗ Status enum not accessible: $e")
    end

    println()
    println("   Testing struct access:")
    try
        Matrix3x3 = getfield(AdvancedTypes, :Matrix3x3)
        println("      ✓ Matrix3x3 struct found")
        println("         - Fields: $(fieldnames(Matrix3x3))")
    catch e
        println("      ✗ Matrix3x3 struct not accessible: $e")
    end

    try
        Grid = getfield(AdvancedTypes, :Grid)
        println("      ✓ Grid struct found")
        println("         - Fields: $(fieldnames(Grid))")
    catch e
        println("      ✗ Grid struct not accessible: $e")
    end

catch e
    println("   ✗ Failed to load wrapper:")
    println("      Error: $e")
    showerror(stdout, e, catch_backtrace())
    println()
end

println()
println("="^70)
println("TEST COMPLETE")
println("="^70)
println()
println("Generated wrapper saved to:")
println("   $wrapper_file")
println()
println("You can inspect the full wrapper with:")
println("   cat $wrapper_file")
println()
