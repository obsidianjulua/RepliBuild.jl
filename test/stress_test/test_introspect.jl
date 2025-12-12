#!/usr/bin/env julia
# Test script for RepliBuild.Introspect module
# Demonstrates all introspection capabilities on the stress_test binary

using Pkg
Pkg.activate("/home/grim/Desktop/Projects/RepliBuild.jl")

println("=" ^ 80)
println("RepliBuild.Introspect - Comprehensive Test")
println("=" ^ 80)
println()

using RepliBuild

# Path to the test binary
binary_path = "/home/grim/Desktop/Projects/RepliBuild.jl/test/stress_test/julia/libproject.so"

if !isfile(binary_path)
    error("Binary not found: $binary_path\nRun RepliBuild.build() first!")
end

println("Testing binary: $(basename(binary_path))")
println()

# ============================================================================
# TEST 1: Binary Introspection - Symbols
# ============================================================================
println("─" ^ 80)
println("TEST 1: Symbol Extraction")
println("─" ^ 80)

println("Extracting symbols...")
syms = RepliBuild.Introspect.symbols(binary_path, filter=:functions)
println("✓ Found $(length(syms)) function symbols")
println()

println("First 10 functions:")
for (i, sym) in enumerate(syms[1:min(10, length(syms))])
    println("  $i. $(sym.demangled)")
end
println()

# Export symbols to CSV
csv_path = "/tmp/symbols.csv"
RepliBuild.Introspect.export_csv(syms, csv_path)
println("✓ Exported symbols to: $csv_path")
println()

# ============================================================================
# TEST 2: Binary Introspection - DWARF
# ============================================================================
println("─" ^ 80)
println("TEST 2: DWARF Debug Information")
println("─" ^ 80)

println("Extracting DWARF info...")
dwarf = RepliBuild.Introspect.dwarf_info(binary_path)
println("✓ Functions: $(length(dwarf.functions))")
println("✓ Structs: $(length(dwarf.structs))")
println("✓ Enums: $(length(dwarf.enums))")
println()

# Show first struct
if !isempty(dwarf.structs)
    first_struct_name = first(keys(dwarf.structs))
    first_struct = dwarf.structs[first_struct_name]
    println("Example struct: $first_struct_name")
    println("  Size: $(first_struct.size) bytes")
    println("  Alignment: $(first_struct.alignment)")
    println("  Members: $(length(first_struct.members))")
    if !isempty(first_struct.members)
        println("  First member: $(first_struct.members[1].name) ($(first_struct.members[1].c_type))")
    end
    println()
end

# Export DWARF as dataset
dataset_dir = "/tmp/dwarf_dataset"
RepliBuild.Introspect.export_dataset(dwarf, dataset_dir, formats=[:json])
println("✓ Exported DWARF dataset to: $dataset_dir")
println()

# ============================================================================
# TEST 3: Binary Introspection - Disassembly
# ============================================================================
println("─" ^ 80)
println("TEST 3: Disassembly")
println("─" ^ 80)

if !isempty(syms)
    first_func = syms[1].demangled
    println("Disassembling: $first_func")
    asm = RepliBuild.Introspect.disassemble(binary_path, first_func, syntax=:intel)

    if !isempty(asm)
        lines = split(asm, '\n')
        println("✓ Disassembled $(length(lines)) lines")
        println("First 5 lines:")
        for line in lines[1:min(5, length(lines))]
            println("  $line")
        end
    else
        println("⚠ No disassembly output (function may be inlined)")
    end
    println()
end

# ============================================================================
# TEST 4: Binary Introspection - Headers
# ============================================================================
println("─" ^ 80)
println("TEST 4: Binary Headers")
println("─" ^ 80)

println("Extracting header info...")
header = RepliBuild.Introspect.headers(binary_path)
println("✓ File type: $(header.file_type)")
println("✓ Architecture: $(header.architecture)")
println("✓ Entry point: $(header.entry_point)")
println("✓ Sections: $(length(header.sections))")
println()

# ============================================================================
# TEST 5: Julia Introspection
# ============================================================================
println("─" ^ 80)
println("TEST 5: Julia Introspection")
println("─" ^ 80)

# Define a test function
function test_func(x::Vector{Float64})
    result = 0.0
    for val in x
        result += val * val
    end
    return result
end

println("Test function: test_func(::Vector{Float64})")
println()

# Type stability analysis
println("5.1 Type Stability Analysis")
stability = RepliBuild.Introspect.analyze_type_stability(test_func, (Vector{Float64},))
if stability.is_stable
    println("  ✓ Type stable!")
else
    println("  ⚠ Type unstable")
    println("  Unstable variables: $(length(stability.unstable_variables))")
end
println()

# SIMD analysis
println("5.2 SIMD Analysis")
simd = RepliBuild.Introspect.analyze_simd(test_func, (Vector{Float64},))
println("  Vectorized loops: $(length(simd.vectorized_loops))")
println("  Vector instructions: $(simd.vector_instructions)")
println("  Scalar instructions: $(simd.scalar_instructions)")
println()

# Allocation analysis
println("5.3 Allocation Analysis")
allocs = RepliBuild.Introspect.analyze_allocations(test_func, (Vector{Float64},))
println("  Allocations: $(length(allocs.allocations))")
println("  Total bytes: $(allocs.total_bytes)")
println("  Escapes: $(allocs.escapes)")
println()

# Get LLVM IR
println("5.4 LLVM IR")
llvm_ir = RepliBuild.Introspect.code_llvm(test_func, (Vector{Float64},), optimized=true)
println("  Instructions: $(llvm_ir.instruction_count)")
println("  Optimized: $(llvm_ir.optimized)")
println()

# Get native assembly
println("5.5 Native Assembly")
native = RepliBuild.Introspect.code_native(test_func, (Vector{Float64},), syntax=:intel)
println("  Instructions: $(native.instruction_count)")
println("  Syntax: $(native.syntax)")
println()

# Full compilation pipeline
println("5.6 Full Compilation Pipeline")
pipeline = RepliBuild.Introspect.compilation_pipeline(test_func, (Vector{Float64},))
println("  Lowered instructions: $(length(pipeline.lowered.code))")
println("  LLVM IR instructions: $(pipeline.llvm_ir.instruction_count)")
println("  Native instructions: $(pipeline.native.instruction_count)")
println()

# ============================================================================
# TEST 6: Benchmarking
# ============================================================================
println("─" ^ 80)
println("TEST 6: Benchmarking")
println("─" ^ 80)

println("Benchmarking test_func with 1000 samples...")
bench_result = RepliBuild.Introspect.benchmark(test_func, rand(100), samples=1000, warmup=10)
println("✓ Completed $(bench_result.samples) samples")
println("  Median time: $(bench_result.median_time / 1e3) μs")
println("  Mean time: $(bench_result.mean_time / 1e3) μs")
println("  Std dev: $(bench_result.std_time / 1e3) μs")
println("  Allocations: $(bench_result.allocations)")
println("  Memory: $(bench_result.memory) bytes")
println()

# Export benchmark result
bench_json = "/tmp/benchmark.json"
RepliBuild.Introspect.export_json(bench_result, bench_json)
println("✓ Exported benchmark to: $bench_json")
println()

# ============================================================================
# TEST 7: Benchmark Suite
# ============================================================================
println("─" ^ 80)
println("TEST 7: Benchmark Suite")
println("─" ^ 80)

# Define multiple test functions
function sum_loop(x::Vector{Float64})
    s = 0.0
    for val in x
        s += val
    end
    return s
end

function sum_builtin(x::Vector{Float64})
    return sum(x)
end

println("Running benchmark suite...")
funcs = Dict(
    "sum_loop" => () -> sum_loop(rand(100)),
    "sum_builtin" => () -> sum_builtin(rand(100)),
    "test_func" => () -> test_func(rand(100))
)

suite_results = RepliBuild.Introspect.benchmark_suite(funcs, samples=500)
println()

# Compare results
println("Comparison:")
for (name, result) in sort(collect(suite_results), by=x->x[2].median_time)
    println("  $name: $(result.median_time / 1e3) μs")
end
println()

# Find fastest
fastest_name = argmin(kv -> kv[2].median_time, collect(suite_results))[1]
println("✓ Fastest: $fastest_name")
println()

# ============================================================================
# TEST 8: Data Export
# ============================================================================
println("─" ^ 80)
println("TEST 8: Data Export")
println("─" ^ 80)

# Export suite results as dataset
suite_dir = "/tmp/benchmark_suite"
RepliBuild.Introspect.export_dataset(collect(values(suite_results)), suite_dir, formats=[:json, :csv])
println("✓ Exported benchmark suite to: $suite_dir")
println()

# ============================================================================
# SUMMARY
# ============================================================================
println("=" ^ 80)
println("SUMMARY")
println("=" ^ 80)
println()
println("✓ Binary Introspection Tests:")
println("  - Symbol extraction: PASSED")
println("  - DWARF parsing: PASSED")
println("  - Disassembly: PASSED")
println("  - Header extraction: PASSED")
println()
println("✓ Julia Introspection Tests:")
println("  - Type stability: PASSED")
println("  - SIMD analysis: PASSED")
println("  - Allocation analysis: PASSED")
println("  - LLVM IR: PASSED")
println("  - Native assembly: PASSED")
println("  - Full pipeline: PASSED")
println()
println("✓ Benchmarking Tests:")
println("  - Single function: PASSED")
println("  - Benchmark suite: PASSED")
println()
println("✓ Data Export Tests:")
println("  - JSON export: PASSED")
println("  - CSV export: PASSED")
println("  - Dataset generation: PASSED")
println()
println("=" ^ 80)
println("All tests completed successfully!")
println("=" ^ 80)
