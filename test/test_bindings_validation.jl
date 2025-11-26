using JSON

# Load and analyze the generated bindings
wrapper_file = "build_advanced/julia/AdvancedTypes.jl"
metadata_file = "build/compilation_metadata.json"

println(repeat("=", 70))
println("BINDING VALIDATION TEST")
println(repeat("=", 70))

# Check wrapper exists
if !isfile(wrapper_file)
    error("Wrapper file not found: $wrapper_file")
end
println("\n✓ Wrapper file exists: $wrapper_file")

# Check metadata exists  
if !isfile(metadata_file)
    error("Metadata file not found: $metadata_file")
end
println("✓ Metadata file exists: $metadata_file")

# Parse metadata
metadata = JSON.parsefile(metadata_file)
println("\n--- Metadata Summary ---")
println("Functions: $(length(metadata["functions"]))")
println("Symbols: $(length(metadata["symbols"]))")
println("Enums: $(length(metadata["enums"]))")

# Analyze binding issues
println("\n--- Binding Analysis ---")

issues = String[]

# Check issue 1: ccall symbol format
wrapper_content = read(wrapper_file, String)
if contains(wrapper_content, "ccall((:matrix_sum(Matrix3x3)")
    push!(issues, "ERROR: ccall using demangled names (should use mangled names)")
end

# Check issue 2: Enum handling
if length(metadata["enums"]) > 0
    enum_check_failed = false
    for (enum_name, enum_info) in metadata["enums"]
        if contains(enum_name, "indexed string")
            enum_check_failed = true
            break
        end
    end
    if enum_check_failed
        push!(issues, "ERROR: Enums not properly parsed (still contain indexed string references)")
    end
end

# Check issue 3: Return type extraction
return_type_issues = 0
for func in metadata["functions"]
    if func["return_type"]["julia_type"] == "Any"
        return_type_issues += 1
    end
end
if return_type_issues > 0
    push!(issues, "WARNING: $(return_type_issues) functions have 'Any' return type (DWARF extraction failed)")
end

# Check issue 4: Parameter type coverage
param_issue_count = 0
for func in metadata["functions"]
    for param in get(func, "parameters", [])
        if param["julia_type"] == "Any"
            param_issue_count += 1
        end
    end
end
if param_issue_count > 0
    push!(issues, "WARNING: $param_issue_count parameters have 'Any' type")
end

println("\nIssues Found: $(length(issues))")
for issue in issues
    println("  - $issue")
end

# Categorize issues by severity
println("\n--- Issue Summary ---")
errors = filter(x -> startswith(x, "ERROR:"), issues)
warnings = filter(x -> startswith(x, "WARNING:"), issues)

println("Critical Errors: $(length(errors))")
for err in errors
    println("  $err")
end

println("Warnings: $(length(warnings))")
for warn in warnings
    println("  $warn")
end

# Overall status
println("\n" * repeat("=", 70))
if length(errors) == 0
    println("RESULT: PASSED (Some warnings present)")
else
    println("RESULT: FAILED (Critical errors present)")
end
println(repeat("=", 70))

# Detailed function analysis
println("\n--- Sample Function Signatures ---")
for func in metadata["functions"][1:min(3, length(metadata["functions"]))]
    println("\n$(func["name"])")
    println("  Demangled: $(func["demangled_name"])")
    println("  Mangled: $(func["mangled_name"])")
    println("  Return: $(func["return_type"]["julia_type"]) (c_type: $(func["return_type"]["c_type"]))")
    for param in get(func, "parameters", [])
        println("  Param: $(param["julia_type"]) (c_type: $(param["c_type"]))")
    end
end

