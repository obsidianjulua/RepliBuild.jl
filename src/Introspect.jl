#!/usr/bin/env julia
# Introspect.jl - Unified Introspection API for RepliBuild
# Provides structured access to binary analysis, Julia introspection, LLVM tooling, and benchmarking
# Designed for dataset generation and performance analysis workflows

module Introspect

# Import parent modules for tool execution and compilation infrastructure
import ..LLVMEnvironment
import ..BuildBridge
import ..Compiler

# Standard library imports
using InteractiveUtils  # For @code_* macros
using Dates            # For timestamps
using Statistics       # For benchmark statistics
using JSON            # For JSON export

# Load submodules in dependency order
include("Introspect/Types.jl")
include("Introspect/DataExport.jl")
include("Introspect/Binary.jl")
include("Introspect/Julia.jl")
include("Introspect/LLVM.jl")
include("Introspect/Benchmarking.jl")

# Re-export public APIs from submodules

# Binary Introspection
export symbols, dwarf_info, disassemble, headers, dwarf_dump

# Julia Introspection
export code_lowered, code_typed, code_llvm, code_native, code_warntype
export analyze_type_stability, analyze_simd, analyze_allocations, analyze_inlining
export compilation_pipeline

# LLVM Tooling
export llvm_ir, optimize_ir, compare_optimization, run_passes, compile_to_asm

# Benchmarking
export benchmark, benchmark_suite, track_allocations, profile

# Dataset Export
export export_json, export_csv, export_dataset

end # module Introspect
