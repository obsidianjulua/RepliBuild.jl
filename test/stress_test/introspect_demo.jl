#!/usr/bin/env julia
# Introspection Demo - Real tooling output on stress_test binary
# Demonstrates RepliBuild.Introspect capabilities on production code

using Pkg
Pkg.activate("/home/grim/Desktop/Projects/RepliBuild.jl")

using RepliBuild
using Printf

println("╔" * "═"^78 * "╗")
println("║" * " "^20 * "RepliBuild Introspection Demo" * " "^28 * "║")
println("║" * " "^15 * "Analyzing stress_test Production Binary" * " "^23 * "║")
println("╚" * "═"^78 * "╝")
println()

# Paths
binary_path = joinpath(@__DIR__, "julia", "libstress_test.so")
wrapper_path = joinpath(@__DIR__, "julia", "StressTest.jl")

if !isfile(binary_path)
    println("❌ Binary not found. Run: RepliBuild.build(\"replibuild.toml\")")
    exit(1)
end

# ============================================================================
# PART 1: Binary Analysis with LLVM Tools
# ============================================================================
println("┏" * "━"^78 * "┓")
println("┃ PART 1: Binary Introspection - What's in libstress_test.so?")
println("┗" * "━"^78 * "┛")
println()

# 1.1 Symbol Table
println("🔍 Symbol Table Analysis (nm wrapper)")
println("─" * "─"^78)
syms = RepliBuild.Introspect.symbols(binary_path, filter=:functions)
println("✓ Found $(length(syms)) exported functions\n")

println("Sample functions (first 15):")
for (i, sym) in enumerate(syms[1:min(15, length(syms))])
    println("  $i. $(sym.demangled)")
    if i == 15
        println("  ... ($(length(syms) - 15) more)")
    end
end
println()

# 1.2 Binary Headers
println("🔍 Binary Header Info (readelf wrapper)")
println("─" * "─"^78)
header = RepliBuild.Introspect.headers(binary_path)
println("File Type:    $(header.file_type)")
println("Architecture: $(header.architecture)")
println("Entry Point:  $(header.entry_point)")
println("Sections:     $(length(header.sections))")
println("\nKey sections:")
for (name, offset, size) in header.sections[1:min(10, length(header.sections))]
    @printf("  %-20s offset: 0x%08x  size: %8d bytes\n", name, offset, size)
end
println()

# 1.3 Disassembly Sample
println("🔍 Disassembly Sample (objdump wrapper)")
println("─" * "─"^78)
if !isempty(syms)
    func_name = syms[1].demangled
    println("Disassembling: $func_name (Intel syntax)")
    asm = RepliBuild.Introspect.disassemble(binary_path, func_name, syntax=:intel)
    if !isempty(asm)
        lines = split(asm, '\n')
        println("\nFirst 20 instructions:")
        for (i, line) in enumerate(lines[1:min(20, length(lines))])
            println("  $line")
        end
        if length(lines) > 20
            println("  ... ($(length(lines) - 20) more lines)")
        end
    end
end
println()

# ============================================================================
# PART 2: DWARF Debug Information
# ============================================================================
println("┏" * "━"^78 * "┓")
println("┃ PART 2: DWARF Debug Info - Type Information from Debug Symbols")
println("┗" * "━"^78 * "┛")
println()

println("🔍 Extracting DWARF Debug Information")
println("─" * "─"^78)
dwarf = RepliBuild.Introspect.dwarf_info(binary_path)
println("Functions: $(length(dwarf.functions))")
println("Structs:   $(length(dwarf.structs))")
println("Enums:     $(length(dwarf.enums))")
println()

# Show struct layouts
if !isempty(dwarf.structs)
    println("📊 Struct Layouts from DWARF:")
    for (i, (name, struct_info)) in enumerate(dwarf.structs)
        if i > 3  # Show first 3 structs
            println("\n  ... ($(length(dwarf.structs) - 3) more structs)")
            break
        end
        println("\n  struct $name {")
        println("    Size: $(struct_info.size) bytes")
        println("    Alignment: $(struct_info.alignment)")
        if !isempty(struct_info.members)
            println("    Members:")
            for member in struct_info.members
                @printf("      +%-4d  %-15s %s\n", member.offset, member.c_type, member.name)
            end
        end
        println("  }")
    end
end
println()

# ============================================================================
# PART 3: Julia Wrapper Analysis
# ============================================================================
println("┏" * "━"^78 * "┓")
println("┃ PART 3: Julia Wrapper Introspection")
println("┗" * "━"^78 * "┛")
println()

if isfile(wrapper_path)
    println("🔍 Analyzing Generated Julia Wrapper")
    println("─" * "─"^78)

    # Load the wrapper
    println("Loading wrapper: $(basename(wrapper_path))")
    include(wrapper_path)

    # Find the module
    if isdefined(Main, :StressTest)
        mod = Main.StressTest
        println("✓ Module loaded: StressTest")
        println()

        # List exported functions
        println("📦 Exported Functions:")
        exported_names = names(mod, all=false)
        for (i, name) in enumerate(exported_names)
            if i <= 20
                println("  $i. $name")
            elseif i == 21
                println("  ... ($(length(exported_names) - 20) more)")
                break
            end
        end
        println()

        # Introspect a wrapped function
        if isdefined(mod, :vector_dot)
            println("🔬 Deep Introspection: vector_dot")
            println("─" * "─"^78)

            func = mod.vector_dot
            println("Function: vector_dot")
            println("Module: StressTest")

            # Get method signatures
            methods_list = methods(func)
            println("\nMethods ($(length(methods_list))):")
            for m in methods_list
                println("  • $m")
            end
            println()
        end
    end
else
    println("⚠ Wrapper not found. Run: RepliBuild.wrap(\"replibuild.toml\")")
end

# ============================================================================
# PART 4: Julia Introspection on Wrapper Functions
# ============================================================================
if isdefined(Main, :StressTest) && isdefined(Main.StressTest, :vector_dot)
    println("┏" * "━"^78 * "┓")
    println("┃ PART 4: Julia Introspection on Wrapped C++ Function")
    println("┗" * "━"^78 * "┛")
    println()

    func = Main.StressTest.vector_dot

    # Create test wrapper for introspection
    function test_wrapper()
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        return Main.StressTest.vector_dot(pointer(a), pointer(b), UInt64(5))
    end

    println("🔬 Analyzing Julia wrapper function")
    println("─" * "─"^78)
    println("Function: test_wrapper() calling vector_dot via ccall")
    println()

    # Type stability
    println("📊 Type Stability Analysis:")
    stability = RepliBuild.Introspect.analyze_type_stability(test_wrapper, ())
    if stability.is_stable
        println("  ✓ Type stable")
    else
        println("  ⚠ Type unstable")
        for var in stability.unstable_variables
            println("    • $(var.variable): $(var.inferred_type)")
        end
    end
    println()

    # LLVM IR
    println("📊 LLVM IR Generation:")
    llvm_info = RepliBuild.Introspect.code_llvm(test_wrapper, (), optimized=true)
    println("  Instructions: $(llvm_info.instruction_count)")
    println("  Optimized: $(llvm_info.optimized)")
    println("\n  Sample IR (first 15 lines):")
    ir_lines = split(llvm_info.ir, '\n')
    for (i, line) in enumerate(ir_lines[1:min(15, length(ir_lines))])
        println("    $line")
    end
    if length(ir_lines) > 15
        println("    ... ($(length(ir_lines) - 15) more lines)")
    end
    println()

    # Native assembly
    println("📊 Native Assembly:")
    native = RepliBuild.Introspect.code_native(test_wrapper, (), syntax=:intel)
    println("  Instructions: $(native.instruction_count)")
    println("  Syntax: $(native.syntax)")
    println("\n  Sample assembly (first 15 lines):")
    asm_lines = split(native.assembly, '\n')
    for (i, line) in enumerate(asm_lines[1:min(15, length(asm_lines))])
        println("    $line")
    end
    if length(asm_lines) > 15
        println("    ... ($(length(asm_lines) - 15) more lines)")
    end
    println()
end

# ============================================================================
# PART 5: Performance Benchmarking
# ============================================================================
if isdefined(Main, :StressTest) && isdefined(Main.StressTest, :vector_dot)
    println("┏" * "━"^78 * "┓")
    println("┃ PART 5: Performance Benchmarking")
    println("┗" * "━"^78 * "┛")
    println()

    println("🏃 Benchmarking C++ Functions via Julia Wrapper")
    println("─" * "─"^78)

    # Benchmark vector_dot
    function bench_vector_dot()
        a = rand(100)
        b = rand(100)
        return Main.StressTest.vector_dot(pointer(a), pointer(b), UInt64(100))
    end

    println("Function: vector_dot (100 elements)")
    result = RepliBuild.Introspect.benchmark(bench_vector_dot, samples=1000, warmup=50)
    println("  Samples:      $(result.samples)")
    println("  Median time:  $(round(result.median_time / 1e3, digits=3)) μs")
    println("  Mean time:    $(round(result.mean_time / 1e3, digits=3)) μs")
    println("  Std dev:      $(round(result.std_time / 1e3, digits=3)) μs")
    println("  Min time:     $(round(result.min_time / 1e3, digits=3)) μs")
    println("  Max time:     $(round(result.max_time / 1e3, digits=3)) μs")
    println("  Allocations:  $(result.allocations)")
    println("  Memory:       $(result.memory) bytes")
    println()

    # Benchmark multiple functions if available
    if isdefined(Main.StressTest, :vector_norm) && isdefined(Main.StressTest, :vector_scale)
        println("📊 Benchmark Suite Comparison")
        println("─" * "─"^78)

        funcs = Dict(
            "vector_dot" => bench_vector_dot,
            "vector_norm" => () -> Main.StressTest.vector_norm(pointer(rand(100)), UInt64(100)),
            "vector_scale" => () -> Main.StressTest.vector_scale(pointer(rand(100)), 2.0, UInt64(100))
        )

        results = RepliBuild.Introspect.benchmark_suite(funcs, samples=500, warmup=20)

        # Sort by speed
        sorted_results = sort(collect(results), by=x->x[2].median_time)
        println("\nResults (fastest to slowest):")
        for (name, res) in sorted_results
            @printf("  %-15s  %8.3f μs  (± %.3f μs)\n",
                    name,
                    res.median_time / 1e3,
                    res.std_time / 1e3)
        end

        fastest = sorted_results[1][1]
        println("\n✓ Fastest: $fastest")
        println()
    end
end

# ============================================================================
# PART 6: Dataset Export
# ============================================================================
println("┏" * "━"^78 * "┓")
println("┃ PART 6: Dataset Generation for Analysis/ML")
println("┗" * "━"^78 * "┛")
println()

println("💾 Exporting Introspection Data")
println("─" * "─"^78)

# Export symbols to CSV
csv_path = "/tmp/replibuild_symbols.csv"
RepliBuild.Introspect.export_csv(syms, csv_path)
println("✓ Symbols exported to: $csv_path")
println("  Format: CSV with columns: name, demangled, address, type, size")

# Export DWARF to JSON
dwarf_path = "/tmp/replibuild_dwarf.json"
RepliBuild.Introspect.export_json(dwarf, dwarf_path)
println("✓ DWARF info exported to: $dwarf_path")
println("  Format: JSON with functions, structs, enums")

# Export complete dataset
dataset_dir = "/tmp/replibuild_dataset"
RepliBuild.Introspect.export_dataset(dwarf, dataset_dir, formats=[:json, :csv])
println("✓ Complete dataset exported to: $dataset_dir/")
println("  Files: functions.json, structs.json, struct_members.csv, etc.")

println()
println("╔" * "═"^78 * "╗")
println("║" * " "^25 * "Introspection Demo Complete" * " "^26 * "║")
println("╚" * "═"^78 * "╝")
println()
println("📝 Summary:")
println("   • Binary analysis with nm, objdump, readelf wrappers")
println("   • DWARF debug info extraction and struct layout visualization")
println("   • Julia wrapper introspection and analysis")
println("   • LLVM IR and native assembly generation")
println("   • Performance benchmarking of C++ functions via Julia")
println("   • Dataset export for ML/analysis workflows")
println()
println("🎯 All introspection features demonstrated on production code!")
