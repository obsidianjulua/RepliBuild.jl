#!/usr/bin/env julia
# test_advanced_extraction.jl
# Test script to verify enum, array, and function pointer extraction from DWARF

using RepliBuild

println("="^70)
println("Testing Advanced Type Extraction (Enums, Arrays, Function Pointers)")
println("="^70)
println()

# Setup test directory
test_dir = @__DIR__
test_cpp = joinpath(test_dir, "test_advanced_types.cpp")
test_so = joinpath(test_dir, "test_advanced_types.so")

println("Step 1: Compiling test C++ file with debug info...")
println("   Source: $test_cpp")

# Compile with clang++
compile_cmd = `clang++ -shared -fPIC -g -O0 -std=c++17 $test_cpp -o $test_so`
println("   Running: $compile_cmd")
run(compile_cmd)

if isfile(test_so)
    println("   ✓ Compiled successfully: $test_so")
else
    error("   ✗ Compilation failed!")
end

println()
println("Step 2: Extracting DWARF debug information...")

# Extract DWARF using RepliBuild's internal function
# We need to access the internal Compiler module
using RepliBuild: Compiler

(return_types, struct_defs) = Compiler.extract_dwarf_return_types(test_so)

println()
println("="^70)
println("EXTRACTION RESULTS")
println("="^70)
println()

# Count extracted types
enum_count = count(k -> startswith(k, "__enum__"), keys(struct_defs))
struct_count = count(k -> !startswith(k, "__enum__") && haskey(struct_defs[k], "members"), keys(struct_defs))

println("📊 Summary:")
println("   Functions: $(length(return_types))")
println("   Structs: $struct_count")
println("   Enums: $enum_count")
println()

# Show enums
if enum_count > 0
    println("🔢 Extracted Enums:")
    for (name, info) in struct_defs
        if startswith(name, "__enum__")
            enum_name = replace(name, "__enum__" => "")
            underlying = get(info, "underlying_type", "int")
            enumerators = get(info, "enumerators", [])
            println("   • $enum_name (underlying: $underlying)")
            for e in enumerators
                println("      - $(e["name"]) = $(e["value"])")
            end
        end
    end
    println()
end

# Show structs with arrays
println("📦 Extracted Structs with Arrays:")
for (name, info) in struct_defs
    if !startswith(name, "__enum__") && haskey(info, "members")
        members = info["members"]
        # Check if any member is an array
        has_array = any(m -> contains(get(m, "c_type", ""), "["), members)
        if has_array
            println("   • $name")
            for m in members
                c_type = get(m, "c_type", "?")
                julia_type = get(m, "julia_type", "?")
                println("      - $(m["name"]): $c_type → $julia_type")
            end
        end
    end
end
println()

# Show function pointers
println("🔗 Function Pointers:")
found_funcptr = false
for (name, info) in struct_defs
    if !startswith(name, "__enum__") && haskey(info, "members")
        members = info["members"]
        for m in members
            c_type = get(m, "c_type", "")
            if contains(c_type, "function_ptr")
                found_funcptr = true
                println("   • $(name).$(m["name"]): $c_type → $(m["julia_type"])")
            end
        end
    end
end
if !found_funcptr
    println("   (None detected - function pointers may be typedef'd)")
end
println()

println("="^70)
println("TEST COMPLETE")
println("="^70)

# Cleanup
rm(test_so, force=true)
println("✓ Cleaned up test artifacts")
