# Zero-Copy Benchmarks

RepliBuild's MLIR `jlcs` dialect allows direct translation of Julia's contiguous memory regions into zero-copy C-struct representations that operate entirely inside JIT boundaries.

To demonstrate the lack of wrapper overhead, we export strict `Introspect.benchmark` data comparing Native Julia Matrix Multiplication (`A * B`) against a naive triple-loop C++ function executing through RepliBuild's zero-copy MLIR dialect.

*(Note: The C++ function is a naive loop, meaning its performance will fall apart at massive sizes due to lacking BLAS vectorization, but it perfectly highlights the function call and wrapper boundary overhead at tiny sizes).*

### Key Takeaway
For a 4x4 matrix, the RepliBuild zero-copy wrapper executes the full JIT context switch and loop logic roughly **35% faster** than Native Julia due to skipping dynamic bounds checking and intermediate array allocations.

It adds effectively zero overhead compared to raw `ccall`, establishing RepliBuild as a zero-cost abstraction for bridging languages.

## 4x4 Benchmark Summary

| System | Median Time | Memory Allocated | Total Allocations |
| ------ | ----------- | ---------------- | ----------------- |
| **Native Julia** | ~540.0 ns | ~200 bytes | 2 |
| **RepliBuild C++ MLIR** | ~400.0 ns | ~144 bytes | 1 |

*For complete datasets on 2x2, 4x4, 16x16, 64x64, and 128x128 matrices, see the `/docs/src/assets/benchmarks` JSON exports.*
