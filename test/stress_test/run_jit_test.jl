#!/usr/bin/env julia
# JIT Integration Test for StressTest
# Tests both Tier 1 (ccall) and Tier 2 (JIT) function dispatch

include(joinpath(@__DIR__, "julia", "StressTest.jl"))
using .StressTest

println("=== Comprehensive JIT Test Suite ===")
println()

# Helper: allocate a DenseMatrix on the heap and return a stable pointer
function heap_alloc(mat::StressTest.DenseMatrix)
    buf = Libc.malloc(sizeof(StressTest.DenseMatrix))
    ptr = Ptr{StressTest.DenseMatrix}(buf)
    unsafe_store!(ptr, mat)
    return ptr
end

# ---- Test 1: dense_matrix_create + operations ----
println("1. Dense Matrix Create (JIT) + Operations")
mat = StressTest.dense_matrix_create(Csize_t(3), Csize_t(3))
mat_ptr = heap_alloc(mat)
StressTest.dense_matrix_set_identity(mat_ptr)
trace = StressTest.matrix_trace(mat_ptr)
@assert trace == 3.0 "Expected trace=3.0, got $trace"
println("   pass: dense_matrix_create -> trace=3.0")

# ---- Test 2: matrix_transpose (JIT, struct return) ----
println("2. Matrix Transpose (JIT)")
mat_t = StressTest.matrix_transpose(mat_ptr)
mat_t_ptr = heap_alloc(mat_t)
trace_t = StressTest.matrix_trace(mat_t_ptr)
@assert trace_t == 3.0 "Expected trace=3.0, got $trace_t"
println("   pass: matrix_transpose -> trace=3.0 (identity transposed)")

# ---- Test 3: matrix_multiply (JIT, struct return) ----
println("3. Matrix Multiply (JIT)")
mat_sq = StressTest.matrix_multiply(mat_ptr, mat_ptr)
mat_sq_ptr = heap_alloc(mat_sq)
trace_sq = StressTest.matrix_trace(mat_sq_ptr)
@assert trace_sq == 3.0 "Expected trace=3.0 (I*I=I), got $trace_sq"
println("   pass: matrix_multiply(I,I) -> trace=3.0")

# ---- Test 4: matrix_add (JIT, struct return) ----
println("4. Matrix Add (JIT)")
mat_sum = StressTest.matrix_add(mat_ptr, mat_ptr)
mat_sum_ptr = heap_alloc(mat_sum)
trace_sum = StressTest.matrix_trace(mat_sum_ptr)
@assert trace_sum == 6.0 "Expected trace=6.0 (I+I=2I), got $trace_sum"
println("   pass: matrix_add(I,I) -> trace=6.0")

# ---- Test 5: dense_matrix_copy (JIT, struct return) ----
println("5. Dense Matrix Copy (JIT)")
mat_copy = StressTest.dense_matrix_copy(mat_ptr)
mat_copy_ptr = heap_alloc(mat_copy)
trace_copy = StressTest.matrix_trace(mat_copy_ptr)
@assert trace_copy == 3.0 "Expected trace=3.0, got $trace_copy"
println("   pass: dense_matrix_copy -> trace=3.0")

# ---- Test 6: Statistical functions (ccall, Tier 1) ----
println("6. Statistical Functions (ccall)")
data = [1.0, 2.0, 3.0, 4.0, 5.0]
mean_val = StressTest.compute_mean(data, Csize_t(5))
@assert mean_val == 3.0 "Expected mean=3.0, got $mean_val"
stddev_val = StressTest.compute_stddev(data, Csize_t(5))
median_val = StressTest.compute_median(data, Csize_t(5))
@assert median_val == 3.0 "Expected median=3.0, got $median_val"
println("   pass: mean=3.0, stddev=$(round(stddev_val, digits=4)), median=3.0")

# ---- Test 7: Vector operations (ccall, Tier 1) ----
println("7. Vector Operations (ccall)")
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
dot = StressTest.vector_dot(pointer(a), pointer(b), Csize_t(3))
@assert dot == 32.0 "Expected dot=32.0, got $dot"
norm_val = StressTest.vector_norm(pointer(a), Csize_t(3))
println("   pass: dot=32.0, norm=$(round(norm_val, digits=4))")

# ---- Cleanup ----
StressTest.dense_matrix_destroy(mat_ptr)
StressTest.dense_matrix_destroy(mat_t_ptr)
StressTest.dense_matrix_destroy(mat_sq_ptr)
StressTest.dense_matrix_destroy(mat_sum_ptr)
StressTest.dense_matrix_destroy(mat_copy_ptr)
# Free the heap-allocated wrappers
for p in [mat_ptr, mat_t_ptr, mat_sq_ptr, mat_sum_ptr, mat_copy_ptr]
    Libc.free(p)
end

println()
println("=== All 7 test groups passed ===")
