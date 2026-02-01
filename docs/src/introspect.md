# Introspection Tools

RepliBuild includes a comprehensive introspection toolkit located in `RepliBuild.Introspect`. This module provides a unified interface for analyzing every stage of the compilation pipeline—from binary artifacts and DWARF debug info to Julia's lowered code and LLVM IR.

## Binary Analysis

These tools allow you to inspect the compiled C++ artifacts directly.

```@docs
RepliBuild.Introspect.symbols
RepliBuild.Introspect.dwarf_info
RepliBuild.Introspect.dwarf_dump
RepliBuild.Introspect.disassemble
RepliBuild.Introspect.headers
```

## Julia Introspection

Analyze how Julia compiles your wrapper code. These functions wrap standard Julia introspection tools but provide more structured output suitable for analysis.

```@docs
RepliBuild.Introspect.code_lowered
RepliBuild.Introspect.code_typed
RepliBuild.Introspect.code_llvm
RepliBuild.Introspect.code_native
RepliBuild.Introspect.code_warntype
RepliBuild.Introspect.analyze_type_stability
RepliBuild.Introspect.analyze_simd
RepliBuild.Introspect.analyze_allocations
RepliBuild.Introspect.analyze_inlining
RepliBuild.Introspect.compilation_pipeline
```

## LLVM Tooling

Work directly with LLVM IR to understand optimization passes and code generation.

```@docs
RepliBuild.Introspect.llvm_ir
RepliBuild.Introspect.optimize_ir
RepliBuild.Introspect.compare_optimization
RepliBuild.Introspect.run_passes
RepliBuild.Introspect.compile_to_asm
```

## Benchmarking

Performance analysis tools designed to compare C++ native performance against Julia wrappers.

```@docs
RepliBuild.Introspect.benchmark
RepliBuild.Introspect.benchmark_suite
RepliBuild.Introspect.track_allocations
```

## Data Export

Export your findings for external analysis or reporting.

```@docs
RepliBuild.Introspect.export_json
RepliBuild.Introspect.export_csv
RepliBuild.Introspect.export_dataset
```

## Real-World Workflow: Analyzing a Slow Function

Suppose you have a wrapped C++ function `compute_physics` that isn't performing as expected. Here is how you can use the introspection toolkit to diagnose the issue.

### 1. Benchmark
First, establish a baseline.

```julia
using RepliBuild.Introspect
using MyWrappedLib

# Run a reliable benchmark
result = benchmark(MyWrappedLib.compute_physics, (data_ptr, 1000))
println("Average time: \$(result.avg_time_ns) ns")
```

### 2. Check Type Stability
If the wrapper is type-unstable, Julia has to box values, killing performance.

```julia
# This will print a warning if return types or variables are not concrete
analyze_type_stability(MyWrappedLib.compute_physics, (data_ptr, 1000))
```

### 3. Inspect Native Code
Did the compiler vectorize the loop? Use `code_native` or `analyze_simd`.

```julia
# Look for vector instructions (e.g., vmovups, vmulpd) in the assembly
analyze_simd(MyWrappedLib.compute_physics, (data_ptr, 1000))
```

### 4. Optimize and Compare
Recompile your library with higher optimization levels using `replibuild.toml` (set `optimization_level = "3"`), then rebuild and compare.

```julia
# Compare the IR of two build variants
compare_optimization("build/O2/lib.so", "build/O3/lib.so")
```
