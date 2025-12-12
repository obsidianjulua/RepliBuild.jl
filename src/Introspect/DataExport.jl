#!/usr/bin/env julia
# DataExport.jl - Export introspection results to JSON and CSV formats
# Provides dataset generation capabilities for ML and analysis workflows

using JSON
using DataFrames
using CSV

# ============================================================================
# JSON EXPORT
# ============================================================================

"""
    export_json(data, path::String; pretty=true)

Export any structured type to JSON format.

# Arguments
- `data` - Data to export (any type with compatible structure)
- `path::String` - Output file path
- `pretty::Bool` - Pretty-print JSON (default: true)

# Examples
```julia
# Export benchmark result
result = benchmark(my_func, args)
export_json(result, "benchmark.json")

# Export DWARF info
dwarf = dwarf_info("lib.so")
export_json(dwarf, "dwarf_info.json")
```
"""
function export_json(data, path::String; pretty::Bool=true)
    # Convert data to JSON-serializable format
    json_data = to_json_dict(data)

    # Write to file
    open(path, "w") do io
        if pretty
            JSON.print(io, json_data, 2)
        else
            JSON.print(io, json_data)
        end
    end

    return path
end

"""Convert various types to JSON-serializable dictionaries"""
function to_json_dict(data)
    # Handle basic types
    if data isa Number || data isa String || data isa Bool || data === nothing
        return data
    end

    # Handle arrays/vectors
    if data isa AbstractArray
        return [to_json_dict(item) for item in data]
    end

    # Handle tuples
    if data isa Tuple
        return [to_json_dict(item) for item in data]
    end

    # Handle dictionaries
    if data isa AbstractDict
        return Dict(string(k) => to_json_dict(v) for (k, v) in data)
    end

    # Handle DateTime
    if data isa DateTime
        return Dates.format(data, "yyyy-mm-dd HH:MM:SS")
    end

    # Handle Type
    if data isa Type
        return string(data)
    end

    # Handle SymbolInfo
    if data isa SymbolInfo
        return Dict(
            "name" => data.name,
            "demangled" => data.demangled,
            "address" => data.address,
            "type" => string(data.type),
            "size" => data.size
        )
    end

    # Handle MemberInfo
    if data isa MemberInfo
        return Dict(
            "name" => data.name,
            "c_type" => data.c_type,
            "julia_type" => data.julia_type,
            "offset" => data.offset,
            "size" => data.size
        )
    end

    # Handle StructInfo
    if data isa StructInfo
        return Dict(
            "name" => data.name,
            "size" => data.size,
            "alignment" => data.alignment,
            "members" => to_json_dict(data.members),
            "base_classes" => data.base_classes,
            "is_polymorphic" => data.is_polymorphic,
            "vtable_offset" => data.vtable_offset
        )
    end

    # Handle FunctionInfo
    if data isa FunctionInfo
        return Dict(
            "name" => data.name,
            "mangled" => data.mangled,
            "demangled" => data.demangled,
            "return_type" => data.return_type,
            "parameters" => [Dict("name" => p[1], "type" => p[2]) for p in data.parameters],
            "is_method" => data.is_method,
            "class" => data.class
        )
    end

    # Handle DWARFInfo
    if data isa DWARFInfo
        return Dict(
            "binary_path" => data.binary_path,
            "functions" => to_json_dict(data.functions),
            "structs" => to_json_dict(data.structs),
            "enums" => Dict(k => [Dict("name" => p[1], "value" => p[2]) for p in v]
                           for (k, v) in data.enums)
        )
    end

    # Handle BenchmarkResult
    if data isa BenchmarkResult
        return Dict(
            "function_name" => data.function_name,
            "samples" => data.samples,
            "median_time" => data.median_time,
            "mean_time" => data.mean_time,
            "std_time" => data.std_time,
            "min_time" => data.min_time,
            "max_time" => data.max_time,
            "allocations" => data.allocations,
            "memory" => data.memory,
            "gc_time" => data.gc_time,
            "timestamp" => to_json_dict(data.timestamp)
        )
    end

    # Handle TypeStabilityAnalysis
    if data isa TypeStabilityAnalysis
        return Dict(
            "function_name" => data.function_name,
            "types" => string(data.types),
            "is_stable" => data.is_stable,
            "unstable_variables" => to_json_dict(data.unstable_variables),
            "warnings" => data.warnings
        )
    end

    # Handle TypeInstability
    if data isa TypeInstability
        return Dict(
            "variable" => data.variable,
            "inferred_type" => string(data.inferred_type),
            "expected_type" => data.expected_type === nothing ? nothing : string(data.expected_type),
            "line" => data.line
        )
    end

    # Handle SIMDAnalysis
    if data isa SIMDAnalysis
        return Dict(
            "function_name" => data.function_name,
            "types" => string(data.types),
            "vectorized_loops" => to_json_dict(data.vectorized_loops),
            "vector_instructions" => data.vector_instructions,
            "scalar_instructions" => data.scalar_instructions
        )
    end

    # Handle VectorizedLoop
    if data isa VectorizedLoop
        return Dict(
            "loop_id" => data.loop_id,
            "vector_width" => data.vector_width,
            "instruction_count" => data.instruction_count
        )
    end

    # Handle AllocationAnalysis
    if data isa AllocationAnalysis
        return Dict(
            "function_name" => data.function_name,
            "types" => string(data.types),
            "allocations" => to_json_dict(data.allocations),
            "total_bytes" => data.total_bytes,
            "escapes" => data.escapes
        )
    end

    # Handle AllocationInfo
    if data isa AllocationInfo
        return Dict(
            "type" => data.type,
            "size" => data.size,
            "line" => data.line
        )
    end

    # Handle OptimizationResult
    if data isa OptimizationResult
        return Dict(
            "ir_path" => data.ir_path,
            "opt_level" => data.opt_level,
            "metrics" => to_json_dict(data.metrics),
            "optimized_ir" => data.optimized_ir
        )
    end

    # Handle OptimizationMetrics
    if data isa OptimizationMetrics
        return Dict(
            "instructions_before" => data.instructions_before,
            "instructions_after" => data.instructions_after,
            "reduction_percentage" => data.reduction_percentage,
            "passes_applied" => data.passes_applied
        )
    end

    # Fallback: try to convert struct fields to dict
    if isstructtype(typeof(data))
        fields = fieldnames(typeof(data))
        return Dict(string(f) => to_json_dict(getfield(data, f)) for f in fields)
    end

    # Last resort: convert to string
    return string(data)
end

# ============================================================================
# CSV EXPORT
# ============================================================================

"""
    export_csv(data::Vector, path::String)

Export vector of structs to CSV format.

# Arguments
- `data::Vector` - Vector of structured data
- `path::String` - Output file path

# Examples
```julia
# Export symbols to CSV
syms = symbols("lib.so")
export_csv(syms, "symbols.csv")

# Export benchmark results
results = [benchmark(f, args) for f in funcs]
export_csv(results, "benchmarks.csv")
```
"""
function export_csv(data::Vector, path::String)
    if isempty(data)
        @warn "Cannot export empty vector to CSV"
        return path
    end

    # Convert to DataFrame
    df = to_dataframe(data)

    # Write to CSV
    CSV.write(path, df)

    return path
end

"""Convert vector of structs to DataFrame"""
function to_dataframe(data::Vector)
    if isempty(data)
        return DataFrame()
    end

    first_item = first(data)

    # Handle SymbolInfo
    if first_item isa SymbolInfo
        return DataFrame(
            name = [d.name for d in data],
            demangled = [d.demangled for d in data],
            address = [d.address for d in data],
            type = [string(d.type) for d in data],
            size = [d.size for d in data]
        )
    end

    # Handle BenchmarkResult
    if first_item isa BenchmarkResult
        return DataFrame(
            function_name = [d.function_name for d in data],
            samples = [d.samples for d in data],
            median_time = [d.median_time for d in data],
            mean_time = [d.mean_time for d in data],
            std_time = [d.std_time for d in data],
            min_time = [d.min_time for d in data],
            max_time = [d.max_time for d in data],
            allocations = [d.allocations for d in data],
            memory = [d.memory for d in data],
            gc_time = [d.gc_time for d in data],
            timestamp = [d.timestamp for d in data]
        )
    end

    # Handle MemberInfo
    if first_item isa MemberInfo
        return DataFrame(
            name = [d.name for d in data],
            c_type = [d.c_type for d in data],
            julia_type = [d.julia_type for d in data],
            offset = [d.offset for d in data],
            size = [d.size for d in data]
        )
    end

    # Fallback: generic struct to DataFrame
    if isstructtype(typeof(first_item))
        fields = fieldnames(typeof(first_item))
        cols = Dict{Symbol,Vector}()

        for field in fields
            values = [getfield(d, field) for d in data]
            # Convert non-primitive types to strings
            if !all(v -> v isa Union{Number,String,Bool,Missing,Nothing} for v in values)
                values = [string(v) for v in values]
            end
            cols[field] = values
        end

        return DataFrame(cols)
    end

    error("Cannot convert $(typeof(first_item)) to DataFrame")
end

# ============================================================================
# DATASET GENERATION
# ============================================================================

"""
    export_dataset(data, dir::String; formats=[:json, :csv])

Export complete dataset in multiple formats.

Organizes data by type and exports to the specified directory.

# Arguments
- `data` - Data to export (DWARFInfo, BenchmarkResult, etc.)
- `dir::String` - Output directory
- `formats::Vector{Symbol}` - Export formats (default: [:json, :csv])

# Examples
```julia
# Export DWARF info as dataset
dwarf = dwarf_info("lib.so")
export_dataset(dwarf, "dataset/")
# Creates: dataset/functions.json, dataset/structs.json, etc.

# Export benchmark suite
results = benchmark_suite(funcs)
export_dataset(results, "benchmarks/", formats=[:json, :csv])
```
"""
function export_dataset(data, dir::String; formats::Vector{Symbol}=[:json, :csv])
    # Create directory if needed
    mkpath(dir)

    # Handle DWARFInfo
    if data isa DWARFInfo
        if :json in formats
            export_json(data.functions, joinpath(dir, "functions.json"))
            export_json(data.structs, joinpath(dir, "structs.json"))
            export_json(data.enums, joinpath(dir, "enums.json"))
        end

        if :csv in formats
            # Export functions
            if !isempty(data.functions)
                func_list = collect(values(data.functions))
                try
                    export_csv(func_list, joinpath(dir, "functions.csv"))
                catch e
                    @warn "Could not export functions to CSV: $e"
                end
            end

            # Export structs
            if !isempty(data.structs)
                struct_list = collect(values(data.structs))
                try
                    # Export struct metadata
                    struct_meta = DataFrame(
                        name = [s.name for s in struct_list],
                        size = [s.size for s in struct_list],
                        alignment = [s.alignment for s in struct_list],
                        is_polymorphic = [s.is_polymorphic for s in struct_list]
                    )
                    CSV.write(joinpath(dir, "structs.csv"), struct_meta)

                    # Export all members
                    all_members = []
                    for s in struct_list
                        for m in s.members
                            push!(all_members, (struct_name=s.name, member=m))
                        end
                    end

                    if !isempty(all_members)
                        members_df = DataFrame(
                            struct_name = [m.struct_name for m in all_members],
                            member_name = [m.member.name for m in all_members],
                            c_type = [m.member.c_type for m in all_members],
                            julia_type = [m.member.julia_type for m in all_members],
                            offset = [m.member.offset for m in all_members],
                            size = [m.member.size for m in all_members]
                        )
                        CSV.write(joinpath(dir, "struct_members.csv"), members_df)
                    end
                catch e
                    @warn "Could not export structs to CSV: $e"
                end
            end
        end
    end

    # Handle Dict of BenchmarkResults
    if data isa Dict && !isempty(data) && first(values(data)) isa BenchmarkResult
        results = collect(values(data))

        if :json in formats
            export_json(data, joinpath(dir, "benchmarks.json"))
        end

        if :csv in formats
            export_csv(results, joinpath(dir, "benchmarks.csv"))
        end
    end

    # Handle Vector of BenchmarkResults
    if data isa Vector && !isempty(data) && first(data) isa BenchmarkResult
        if :json in formats
            export_json(data, joinpath(dir, "benchmarks.json"))
        end

        if :csv in formats
            export_csv(data, joinpath(dir, "benchmarks.csv"))
        end
    end

    # Handle Vector of SymbolInfo
    if data isa Vector && !isempty(data) && first(data) isa SymbolInfo
        if :json in formats
            export_json(data, joinpath(dir, "symbols.json"))
        end

        if :csv in formats
            export_csv(data, joinpath(dir, "symbols.csv"))
        end
    end

    return dir
end
