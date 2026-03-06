## Per-Call Overhead (vs Bare `ccall`)

| Scenario | Tier | Median (ns) | Vs Bare `ccall` | Note |
| :--- | :--- | :--- | :--- | :--- |
| `scalar_add` | `pure_julia` | 30.0 | 1.0x | Julia native `a+b` |
| `scalar_add` | `bare_ccall` | 30.0 | 1.0x | Hand-written `ccall` (community baseline) |
| `scalar_add` | `wrapper_ccall` | 30.0 | 1.0x | RepliBuild generated `ccall` wrapper |
| `scalar_add` | **`lto_llvmcall`** | **30.0** | **1.0x** | **RepliBuild LTO: `Base.llvmcall` (Julia JIT inlines C++ IR)** |
| `scalar_mul` | `pure_julia` | 40.0 | 1.0x | Julia native `a*b` |
| `scalar_mul` | `bare_ccall` | 40.0 | 1.0x | Hand-written `ccall` |
| `scalar_mul` | `wrapper_ccall` | 60.0 | 1.5x | RepliBuild generated `ccall` wrapper |
| `scalar_mul` | **`lto_llvmcall`** | **40.0** | **1.0x** | **RepliBuild LTO: `Base.llvmcall`** |
| `make_point` | `pure_julia` | 40.0 | 1.0x | Julia native struct construction |
| `make_point` | `bare_ccall` | 40.0 | 1.0x | Hand-written `ccall` (struct return, manual layout) |
| `make_point` | `wrapper_ccall` | 40.0 | 1.0x | RepliBuild generated wrapper (ABI-verified layout) |
| `make_point` | **`lto_llvmcall`** | **40.0** | **1.0x** | **RepliBuild LTO: `Base.llvmcall`** |
| `pack_record`| `bare_ccall_UNSAFE` | NaN | — | ⚠ Naive `ccall` — packed struct return crashes; cannot safely benchmark |
| `pack_record`| `wrapper_ccall` | 80.0 | — | RepliBuild generated wrapper (DWARF-verified packed layout) |

---

## Hot Loop (1,000,000 Iterations)
*Running `add_to(acc, val)` continuously across the FFI boundary.*

| Tier | Median (ns) | ns / iter | Note |
| :--- | :--- | :--- | :--- |
| `pure_julia` | 676,784.0 | **0.677** | Julia `@inbounds` loop with native add |
| `bare_ccall_loop` | 1,800,660.0 | **1.801** | Julia loop — bare `ccall` in a typed function |
| `wrapper_ccall_loop` | 676,889.0 | **0.677** | Julia loop calling RepliBuild `ccall` wrapper each iteration |
| **`lto_llvmcall_loop`** | **676,624.0** | **0.677** | **Julia loop with LTO: Julia JIT inlines C++ `add_to` across FFI boundary** |
| `whole_loop_in_cpp` | 996,968.0 | **0.997** | Single `ccall` to C++ `accumulate_array` (entire loop in C++) |