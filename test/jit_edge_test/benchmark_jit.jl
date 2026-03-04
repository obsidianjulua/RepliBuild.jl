# JIT Tier Benchmark: Tier 0 (bare ccall) vs Tier 1 (wrapper) vs Tier 2 (JIT thunk)
# Measures per-call overhead of each dispatch path

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.add("BenchmarkTools")

using RepliBuild
using BenchmarkTools
using Libdl

println("="^70)
println("Building JIT Edge Test for Benchmarking...")
println("="^70)

toml_path = joinpath(@__DIR__, "replibuild.toml")
if !isfile(toml_path)
    RepliBuild.discover(@__DIR__, force=true, build=true, wrap=true)
else
    RepliBuild.build(toml_path)
    RepliBuild.wrap(toml_path)
end

include(joinpath(@__DIR__, "julia", "JitEdgeTest.jl"))
using .JitEdgeTest

# Load library for bare ccall
lib_path = joinpath(@__DIR__, "julia", "libjit_edge_test.so")
lib_handle = Libdl.dlopen(lib_path)

sym_scalar_add = Libdl.dlsym(lib_handle, :scalar_add)
sym_scalar_mul = Libdl.dlsym(lib_handle, :scalar_mul)
sym_identity = Libdl.dlsym(lib_handle, :identity)
sym_make_pair = Libdl.dlsym(lib_handle, :make_pair)

println("\n" * "="^70)
println("JIT Dispatch Tier Benchmarks")
println("="^70)

# ============================================================================
# scalar_add(int, int) -> int
# ============================================================================
println("\n--- scalar_add(int, int) -> int ---")

println("\nTier 0: Bare ccall")
b0 = @benchmark ccall($sym_scalar_add, Cint, (Cint, Cint), Cint(3), Cint(7))
display(b0)

println("\n\nTier 1: Generated wrapper (ccall)")
b1 = @benchmark JitEdgeTest.scalar_add(Cint(3), Cint(7))
display(b1)

println("\n\nTier 2: JIT thunk (MLIR → LLVM → invoke)")
b2 = @benchmark RepliBuild.JITManager.invoke("_mlir_ciface_scalar_add_thunk", Cint, Cint(3), Cint(7))
display(b2)

# ============================================================================
# scalar_mul(double, double) -> double
# ============================================================================
println("\n\n--- scalar_mul(double, double) -> double ---")

println("\nTier 0: Bare ccall")
b0f = @benchmark ccall($sym_scalar_mul, Cdouble, (Cdouble, Cdouble), 2.5, 4.0)
display(b0f)

println("\n\nTier 1: Generated wrapper (ccall)")
b1f = @benchmark JitEdgeTest.scalar_mul(2.5, 4.0)
display(b1f)

println("\n\nTier 2: JIT thunk")
b2f = @benchmark RepliBuild.JITManager.invoke("_mlir_ciface_scalar_mul_thunk", Cdouble, Cdouble(2.5), Cdouble(4.0))
display(b2f)

# ============================================================================
# identity(int) -> int (minimum call overhead)
# ============================================================================
println("\n\n--- identity(int) -> int (minimum overhead) ---")

println("\nTier 0: Bare ccall")
b0i = @benchmark ccall($sym_identity, Cint, (Cint,), Cint(42))
display(b0i)

println("\n\nTier 1: Generated wrapper")
b1i = @benchmark JitEdgeTest.identity(Cint(42))
display(b1i)

println("\n\nTier 2: JIT thunk")
b2i = @benchmark RepliBuild.JITManager.invoke("_mlir_ciface_identity_thunk", Cint, Cint(42))
display(b2i)

# ============================================================================
# make_pair(int, int) -> PairResult (struct return)
# ============================================================================
println("\n\n--- make_pair(int, int) -> PairResult (struct return) ---")

println("\nTier 0: Bare ccall")
b0p = @benchmark ccall($sym_make_pair, JitEdgeTest.PairResult, (Cint, Cint), Cint(11), Cint(22))
display(b0p)

println("\n\nTier 1: Generated wrapper")
b1p = @benchmark JitEdgeTest.make_pair(Cint(11), Cint(22))
display(b1p)

println("\n\nTier 2: JIT thunk")
b2p = @benchmark RepliBuild.JITManager.invoke("_mlir_ciface_make_pair_thunk", JitEdgeTest.PairResult, Cint(11), Cint(22))
display(b2p)

# ============================================================================
# pack_three — packed struct (JIT only, Tier 2)
# ============================================================================
println("\n\n--- pack_three(char, int, char) -> PackedTriplet (packed struct, JIT only) ---")

println("\nTier 2: JIT thunk (packed struct marshalling)")
b2pack = @benchmark JitEdgeTest.pack_three(UInt8('A'), Cint(999), UInt8('Z'))
display(b2pack)

# ============================================================================
# Summary Table
# ============================================================================
println("\n\n" * "="^70)
println("Summary (median times)")
println("="^70)
println("Function          | Tier 0 (bare) | Tier 1 (wrapper) | Tier 2 (JIT)")
println("-"^70)

fmt(b) = "$(round(median(b).time, digits=1)) ns"

println("scalar_add        | $(fmt(b0))   | $(fmt(b1))     | $(fmt(b2))")
println("scalar_mul        | $(fmt(b0f))   | $(fmt(b1f))     | $(fmt(b2f))")
println("identity          | $(fmt(b0i))   | $(fmt(b1i))     | $(fmt(b2i))")
println("make_pair (struct) | $(fmt(b0p))   | $(fmt(b1p))     | $(fmt(b2p))")
println("pack_three (packed)| —             | —                | $(fmt(b2pack))")

# ============================================================================
# JIT Pipeline Overhead (one-time cost)
# ============================================================================
println("\n\n" * "="^70)
println("JIT Pipeline Overhead (one-time initialization cost)")
println("="^70)

# Measure each phase individually
using RepliBuild.MLIRNative
using RepliBuild.JLCSIRGenerator
using RepliBuild.DWARFParser

println("\n1. DWARF vtable parsing")
t_dwarf = @elapsed vtinfo = DWARFParser.parse_vtables(lib_path)
println("   $(round(t_dwarf * 1000, digits=2)) ms")

println("\n2. MLIR IR generation")
metadata = JSON.parsefile(joinpath(@__DIR__, "julia", "compilation_metadata.json"))
t_irgen = @elapsed ir_source = JLCSIRGenerator.generate_jlcs_ir(vtinfo, metadata)
println("   $(round(t_irgen * 1000, digits=2)) ms")

println("\n3. MLIR context creation")
t_ctx = @elapsed ctx = MLIRNative.create_context()
println("   $(round(t_ctx * 1000, digits=2)) ms")

println("\n4. MLIR module parsing")
t_parse = @elapsed mod = MLIRNative.parse_module(ctx, ir_source)
println("   $(round(t_parse * 1000, digits=2)) ms")

println("\n5. JLCS → LLVM lowering")
t_lower = @elapsed MLIRNative.lower_to_llvm(mod)
println("   $(round(t_lower * 1000, digits=2)) ms")

println("\n6. JIT engine creation")
t_jit = @elapsed jit = MLIRNative.create_jit(mod, opt_level=3, shared_libs=[lib_path])
println("   $(round(t_jit * 1000, digits=2)) ms")

total = t_dwarf + t_irgen + t_ctx + t_parse + t_lower + t_jit
println("\nTotal JIT pipeline: $(round(total * 1000, digits=2)) ms")

MLIRNative.destroy_jit(jit)
MLIRNative.destroy_context(ctx)

import JSON

println()
