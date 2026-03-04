using RepliBuild
using RepliBuild.Compiler: extract_dwarf_return_types, extract_symbols_from_binary

binary_path = "test/hello_world_test/julia/libhello_world_test.so"

println("Analyzing binary: $binary_path")
println("="^70)

if !isfile(binary_path)
    println("Error: Binary not found at $binary_path")
    exit(1)
end

# 1. Run readelf manually to see raw output
println("1. Raw readelf output (first 100 lines):")
println("-"^70)
readelf_cmd = `readelf --debug-dump=info $binary_path`
try
    output = read(readelf_cmd, String)
    lines = split(output, '\n')
    for (i, line) in enumerate(lines)
        if i > 100
            break
        end
        println(line)
    end
catch e
    println("Error running readelf: $e")
end
println("-"^70)

# 2. Check symbol extraction
println("\n2. Symbol Extraction:")
symbols = extract_symbols_from_binary(binary_path)
for sym in symbols
    println("  Symbol: $(sym["demangled"]) (Mangled: $(sym["mangled"]))")
end

# 3. Check DWARF return type extraction
println("\n3. DWARF Return Type Extraction:")
try
    (return_types, struct_defs) = extract_dwarf_return_types(binary_path)
    
    println("\n  Found $(length(return_types)) return types from DWARF")
    for (mangled, info) in return_types
        println("  Function: $mangled")
        println("    Return Type: $(get(info, "c_type", "unknown"))")
    end

    if isempty(return_types)
        println("  No return types found! Debugging parsing logic...")
    end
    
catch e
    println("Error in extract_dwarf_return_types: $e")
    Base.show_backtrace(stdout, catch_backtrace())
end
