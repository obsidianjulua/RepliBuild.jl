# Rust Generator Exploration

**Date**: 2026-03-09
**Status**: Exploratory — pipeline proven, generator not yet written

## Why Rust

C++ ABI is a bottomless pit: name mangling, STL internal type leakage through DWARF,
template instantiation stubs, virtual dispatch, exception propagation, allocator-aware
containers. RepliBuild's C++ generator handles the practical 80%, but the remaining 20%
yields diminishing returns.

Rust's `extern "C"` ABI is clean by design. The entire RepliBuild pipeline
(compile → DWARF introspect → wrap) transfers with minimal changes.

## Toolchain

```
rustc 1.93.1 (Arch Linux) → LLVM 21.1.8 (same as system LLVM)
```

`rustc` emits:
- **Shared libraries** (`--crate-type cdylib`) — standard `.so` with clean symbol table
- **LLVM IR** (`--emit=llvm-ir`) — text IR for inspection
- **LLVM bitcode** (`--emit=llvm-bc`) — for LTO / Tier 1 `Base.llvmcall`
- **DWARF debug info** (`-g`) — struct layouts, enum values, function signatures

Since Arch's `rustc` uses the same LLVM 21 as the system, bitcode is version-compatible.
No IR sanitization needed for the Rust→Julia LTO path (unlike C++ where system
LLVM 21 IR must be downgraded for Julia's internal LLVM 18).

## DWARF Analysis

### What Rust DWARF gives us (that C++ doesn't)

| Feature | C++ DWARF | Rust DWARF |
|---------|-----------|------------|
| Symbol names | Mangled (`_Z7sum_vecRKSt6vectorIiSaIiEE`) | Clean (`sum_array`) |
| Struct fields | Mixed with STL internals | Only user-defined types |
| Enum values | Rare / compiler-dependent | First-class with `DW_TAG_enumerator` |
| Type names | Nested template soup | `lib::Point`, `lib::Color` |
| Namespace | `std::__cxx11::basic_string<...>` | `lib::Point` (module path) |
| Internal types | Hundreds (allocator, char_traits, _Vector_impl...) | None leaked to user API |

### Struct example: `Point`

```
DW_TAG_structure_type
  DW_AT_name       ("Point")
  DW_AT_byte_size  (0x10)
  DW_AT_alignment  (8)

  DW_TAG_member
    DW_AT_name     ("x")
    DW_AT_type     (f64)
    DW_AT_data_member_location (0x00)

  DW_TAG_member
    DW_AT_name     ("y")
    DW_AT_type     (f64)
    DW_AT_data_member_location (0x08)
```

Maps directly to Julia:
```julia
struct Point
    x::Cdouble
    y::Cdouble
end
```

### Enum example: `Color`

```
DW_TAG_enumeration_type
  DW_AT_name       ("Color")
  DW_AT_byte_size  (0x04)

  DW_TAG_enumerator  DW_AT_name("Red")    DW_AT_const_value(0)
  DW_TAG_enumerator  DW_AT_name("Green")  DW_AT_const_value(1)
  DW_TAG_enumerator  DW_AT_name("Blue")   DW_AT_const_value(2)
```

Maps directly to Julia:
```julia
@enum Color::Cint Red=0 Green=1 Blue=2
```

### Function signatures (LLVM IR)

```llvm
define i32 @add(i32 %a, i32 %b)                          ; scalar → Tier 1 or 3
define { double, double } @point_new(double %x, double %y) ; small struct return in regs
define double @point_distance(ptr align 8 %a, ptr align 8 %b) ; ref params → just pointers
define ptr @greet(ptr %name)                               ; heap alloc, caller frees
define i64 @sum_array(ptr %data, i64 %len)                 ; slice pattern
define ptr @counter_new(i64 %initial)                      ; opaque type
```

Key observations:
- **No mangling**: symbols are exactly `add`, `point_new`, `counter_new`
- **Small structs in registers**: `Point` (16 bytes) returned as `{ double, double }`
- **References = pointers**: `&Point` becomes `ptr align 8`
- **Enums = integers**: `Color` is `i32`
- **No hidden internal types**: zero STL-like soup in DWARF

## Rust Primitive Type Mapping

| Rust | DWARF | LLVM IR | Julia |
|------|-------|---------|-------|
| `i8` | `i8` | `i8` | `Int8` |
| `u8` | `u8` | `i8` | `UInt8` |
| `i16` | `i16` | `i16` | `Int16` |
| `u16` | `u16` | `i16` | `UInt16` |
| `i32` | `i32` | `i32` | `Cint` / `Int32` |
| `u32` | `u32` | `i32` | `Cuint` / `UInt32` |
| `i64` | `i64` | `i64` | `Int64` |
| `u64` | `u64` | `i64` | `UInt64` |
| `f32` | `f32` | `float` | `Cfloat` / `Float32` |
| `f64` | `f64` | `double` | `Cdouble` / `Float64` |
| `bool` | `bool` | `i1` (zeroext) | `Bool` |
| `usize` | `usize` | `i64` (on 64-bit) | `Csize_t` |
| `isize` | `isize` | `i64` | `Cssize_t` |
| `*const T` | `T *` | `ptr` | `Ptr{T}` |
| `*mut T` | `T *` | `ptr` | `Ptr{T}` |
| `&T` | reference | `ptr align N` | `Ref{T}` or `Ptr{T}` |
| `()` | `()` | `void` | `Cvoid` / `Nothing` |
| `*const c_char` | `i8 *` | `ptr` | `Cstring` or `Ptr{UInt8}` |

## What the Generator Needs

### Reusable as-is (no changes needed)

- `DWARFParser.jl` — struct/enum extraction works identically
- `JITManager.jl` — thunk dispatch unchanged
- `MLIRNative.jl` / JLCS dialect — language-agnostic
- `STLWrappers.jl` — not needed for Rust (no STL)
- `PackageRegistry.jl` — registration unchanged
- `ConfigurationManager.jl` — add `language = "rust"` option

### Needs Rust-specific implementation

| Component | Work needed |
|-----------|-------------|
| `Compiler.jl` | Add `rustc` compilation path: `rustc --crate-type cdylib --emit=llvm-bc,link -g` |
| `Discovery.jl` | Detect `.rs` files, find `Cargo.toml` or standalone crate roots |
| `Wrapper/Rust/GeneratorRust.jl` | New file — the Rust wrapper generator |
| `Wrapper/Rust/TypesRust.jl` | Rust primitive mapping + `lib::` prefix stripping |

### Generator complexity estimate

| Feature | C Generator | C++ Generator | Rust Generator (est.) |
|---------|------------|---------------|----------------------|
| Struct layout | DWARF → Julia struct | Same + packed/blob fallback | Same as C (repr(C) is C-compatible) |
| Enums | Regex + AST | Same | DWARF-native (cleaner) |
| Function dispatch | 3-tier | 3-tier + STL handling | 3-tier (simpler — no STL) |
| Type resolution | ~200 lines | ~400 lines (templates, STL) | ~150 lines (no templates) |
| Name sanitization | Minimal | Heavy (mangling, templates) | Minimal (clean symbols) |
| Opaque types | Forward decl | Forward decl + STL wrappers | Forward decl only |
| **Total estimate** | ~2000 lines | ~2800 lines | **~1200 lines** |

## Tier 1 LTO Opportunity

Since `rustc` uses LLVM 21 (same as system), and emits `.bc` bitcode:

```
rustc --emit=llvm-bc  →  .bc file  →  Base.llvmcall((BC, "add"), Cint, Tuple{Cint, Cint}, a, b)
```

For pure scalar functions like `add(i32, i32) -> i32`, this enables cross-language
inlining — Julia's JIT can inline Rust code. Same Tier 1 path as C with LTO enabled.

**Caveat**: Rust bitcode includes panic infrastructure, allocator calls, etc. for non-trivial
functions. The sanitizer may need to strip `personality` clauses and `landingpad` blocks
(Rust exception handling) before feeding to Julia's LLVM. This is a future optimization —
Tier 3 ccall works out of the box for everything.

## Opaque Types

Rust types without `#[repr(C)]` are opaque — their layout is not ABI-stable. These
appear in DWARF but should NOT be mapped to Julia structs. The generator should:

1. Detect types only accessed through `*mut T` / `*const T` (never by value)
2. Emit them as opaque forward declarations: `struct Counter end`
3. Use `Ptr{Cvoid}` in function signatures (same as C++ opaque types)

The DWARF `DW_AT_accessibility` attribute helps: private fields suggest opaque ownership.

## Test Verification

`test/rust_test/verify.jl` — 21 tests covering:

| Category | Tests | Pattern |
|----------|-------|---------|
| Scalars | 5 | `add`, `multiply_f64`, `is_positive` |
| Structs | 4 | `point_new`, `point_distance`, `rect_area` |
| Strings | 3 | `string_length`, `greet` + `free_string` |
| Enums | 3 | `color_name` for Red/Green/Blue |
| Arrays | 2 | `sum_array` with data + null safety |
| Opaque | 4 | `counter_new/increment/get/free` lifecycle |

All 21 pass in 0.1 seconds via hand-written ccall.

## Next Steps

1. **`TypesRust.jl`** — Rust primitive type mapping (the table above)
2. **`GeneratorRust.jl`** — Wrapper generator, starting with Tier 3 ccall for everything
3. **`Compiler.jl` extension** — `rustc` compilation path alongside clang
4. **`Discovery.jl` extension** — Detect Rust crates, generate `replibuild.toml`
5. **LTO exploration** — Try feeding Rust `.bc` through the existing bitcode pipeline
6. **Cargo integration** — Support `Cargo.toml`-based projects (not just standalone `.rs`)
