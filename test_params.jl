#!/usr/bin/env julia
# Quick test to check parameter extraction works

using RepliBuild
using RepliBuild.Compiler

binary = "test/build_advanced/libadvanced_types.so"

println("Testing parameter extraction on: $binary")
println("="^70)

# Use exported function from Compiler module
(types, enums) = RepliBuild.Compiler.extract_dwarf_return_types(binary)

# Find grid_get function (should have 3 parameters: Grid g, int row, int col)
for (name, info) in types
    if contains(name, "grid_get")
        println("\n✓ Found function: $name")
        println("  Return type: ", get(info, "c_type", "unknown"), " → ", get(info, "julia_type", "Any"))

        if haskey(info, "parameters")
            params = info["parameters"]
            println("  Parameters: ", length(params))
            if isempty(params)
                println("    ❌ EMPTY - extraction failed!")
            else
                for (i, p) in enumerate(params)
                    println("    [$i] ", get(p, "name", "unnamed"), ": ",
                           get(p, "c_type", "unknown"), " → ",
                           get(p, "julia_type", "Any"))
                end
            end
        else
            println("  ❌ NO PARAMETERS FIELD!")
        end
        break
    end
end

println("\n" * "="^70)
