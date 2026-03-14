# Why RepliBuild

## The problem

Calling C/C++ from Julia means writing `ccall` by hand. For a simple function that's fine — but real libraries have hundreds of functions, packed structs, bitfields, unions, virtual methods, and platform-dependent alignment rules. Getting any of those wrong produces silent memory corruption or segfaults with no diagnostics.

Existing tools each solve part of this:

- **Manual `ccall`** gives full control but requires you to manually specify every struct layout, field offset, and calling convention. One wrong `Cint` vs `Clong` and your struct reads garbage.
- **CxxWrap.jl** handles C++ idioms (exceptions, STL) but requires writing and maintaining a separate C++ wrapper layer. The wrapper is another codebase to keep in sync.
- **Clang.jl** parses headers via libclang and can generate bindings from the AST. Good for C, but headers alone don't capture what the compiler actually did — padding, alignment, packed attribute effects, vtable layout.

RepliBuild takes a different approach: compile the source, then extract what the compiler knows.

## How it works

RepliBuild compiles your C/C++ source with Clang (`-g` for debug info), then combines multiple information sources to generate bindings that are correct by construction:

**DWARF debug metadata** (the backbone):
- Struct member offsets, sizes, and byte positions — exactly as the compiler laid them out
- Function return types and parameter types
- Enum values and underlying types
- Class inheritance and virtual method information
- Bitfield widths and positions

**Symbol table** (`nm`):
- Mangled C++ names — the authoritative linking identity
- Function addresses in the binary
- Vtable addresses for virtual dispatch

**Clang.jl AST** (fills gaps):
- Enums that the compiler optimized away (unused in compiled code, but needed for bindings)
- Function pointer typedefs
- Macro definitions (via typed shims declared in `[wrap.macros]`)

**Cross-verification**:
- DWARF struct size vs Julia alignment calculation — detects packed structs
- Packed structs get routed to MLIR thunks instead of `ccall` (which would silently misalign)
- Symbol names validate DWARF completeness; if both exist, DWARF wins

This is why RepliBuild catches things that header parsing alone cannot. When a struct is `#pragma pack(1)`, the header looks identical — but DWARF records `member b at offset 1` instead of `offset 4`. RepliBuild sees this, flags the struct as packed, and generates a safe MLIR thunk instead of a broken `ccall`.

## Three-tier dispatch

Every function gets automatically routed to the right calling mechanism:

| Tier | Mechanism | When | Cost |
|------|-----------|------|------|
| **1** | `Base.llvmcall` with LTO bitcode | POD args, scalar/pointer return, bitcode available | **Zero** — C++ inlines into Julia's JIT, cross-language optimization |
| **2** | MLIR thunks (`libJLCS.so`) | Packed structs, unions, large struct return, virtual dispatch | ~40 ns per call; first call ~500 µs for JIT (or zero with AOT) |
| **3** | `ccall` | Fallback when bitcode unavailable | ~40 ns per call |

You don't choose tiers. RepliBuild analyses each function's DWARF metadata, checks if `ccall` is safe (alignment, size, type complexity), and emits the appropriate wrapper. If the struct layout doesn't match what Julia would expect, it goes to MLIR. If LTO bitcode is available and the types are simple, it goes straight through `llvmcall` with zero FFI overhead.

## When to use RepliBuild

**Use RepliBuild when:**
- You have C/C++ **source code** (not just a pre-compiled binary)
- You want ABI-correct bindings without manual struct definitions
- You need C++ virtual method dispatch from Julia
- You want LTO — C++ code inlined into Julia loops, Enzyme-compatible AD
- Your library uses packed structs, bitfields, unions, or complex types

**Pre-compiled binaries work too:** If you have a `.so`/`.dylib` compiled with debug info (`-g`), RepliBuild can wrap it directly — skip `discover()`, point your `replibuild.toml` at the binary, and run `wrap()`. No source code needed, as long as DWARF metadata is present in the binary.

**Use something else when:**
- You have a stripped binary with no debug info and no source → manual `ccall` or Clang.jl header parsing
- You need C++ exception handling or full STL integration → CxxWrap.jl
- Your library is a single C function with two `int` args → just write the `ccall`

## Comparison

| | RepliBuild | CxxWrap.jl | Clang.jl | Manual ccall |
|---|---|---|---|---|
| **Source needed** | No — source or debug binary | Yes (C++ wrapper layer) | No (headers only) | No |
| **Struct layouts** | DWARF — compiler-verified | Manual C++ definitions | Header parsing — may miss padding | Manual — error-prone |
| **Packed structs** | Auto-detected, MLIR thunks | Manual handling | Not detected | Silent corruption |
| **Virtual methods** | MLIR JIT/AOT thunks | Supported natively | Not supported | Not practical |
| **LTO / inlining** | Yes (`llvmcall` with bitcode) | No | No | No |
| **Maintenance** | `replibuild.toml` — rebuild on source change | Separate C++ wrapper codebase | Re-run generator | Manual updates |
| **Enums** | DWARF + Clang.jl AST (catches optimized-away enums) | Manual | Header parsing | Manual |
| **Macros** | Typed shims via `[wrap.macros]` | Not applicable | Can extract | Manual |

## What makes it different

Most binding generators work from declarations (headers, IDL files). RepliBuild works from the compiled artifact. The compiler has already resolved every `#ifdef`, computed every struct layout, applied every `__attribute__((packed))`, and recorded it in DWARF. RepliBuild reads that and generates Julia code that matches exactly.

The MLIR dialect (`jlcs`) exists because some ABI situations can't be handled by `ccall` at all — packed struct field access, virtual method dispatch through vtables, strided array views. Rather than emit fragile inline LLVM IR, RepliBuild lowers these through a verified MLIR pipeline that produces correct machine code for the target platform. The dialect is small (6 operations) and purpose-built for the Julia↔C++ boundary.

Templates, varargs, and macros don't appear in DWARF by default. RepliBuild handles these through configuration:

```toml
[types]
templates = ["std::vector<int>"]       # Forces Clang to emit DWARF for these instantiations
template_headers = ["<vector>"]

[wrap.varargs]
printf = [["Cstring", "Cint"], ["Cstring", "Cdouble"]]  # Typed overloads

[wrap.macros]
MAX_SIZE = { type = "Cint" }           # Generates typed shim function
```

This is the trade-off: RepliBuild gives you correct, verified, optimizable bindings — but you need either source code or a debug-compiled binary, and an LLVM 21+ toolchain. With source you get the full pipeline (LTO, AOT thunks, incremental builds). With a debug binary you get DWARF-correct wrappers and MLIR thunks. Either way, it eliminates the entire class of ABI bugs that make FFI painful.
