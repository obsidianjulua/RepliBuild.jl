# RepliBuild.jl — Architecture

> ABI-aware C/C++ compiler bridge for Julia, powered by MLIR.

RepliBuild compiles C/C++ source through an LLVM/MLIR pipeline, introspects DWARF debug metadata, and emits type-safe Julia bindings with correct struct layout, enum definitions, and calling conventions. Functions requiring non-trivial ABI handling are automatically routed through a custom MLIR dialect and JIT tier.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          User API (3 functions)                         │
│                                                                         │
│    discover("path/")          build("replibuild.toml")                  │
│    ─── scan & configure ───   ─── compile & link ───                    │
│                                                                         │
│                               wrap("replibuild.toml")                   │
│                               ─── introspect & emit Julia module ───    │
└─────────────┬───────────────────────────┬───────────────────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────────┐   ┌───────────────────────────────────────────┐
│    Configuration Layer   │   │            Compiler Pipeline              │
│                          │   │                                           │
│  Discovery.jl            │   │  Compiler.jl → BuildBridge.jl → Linker   │
│  ConfigurationManager.jl │   │  DependencyResolver.jl                   │
│  LLVMEnvironment.jl      │   │  LLVMEnvironment.jl                      │
│  EnvironmentDoctor.jl    │   │                                           │
└─────────────┬───────────┘   └──────────────────┬────────────────────────┘
              │                                   │
              │         replibuild.toml           │  .so + DWARF + .ll
              ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Binding Generation                               │
│                                                                         │
│  DWARFParser.jl ──→ Wrapper.jl ──→ Generated Julia Module               │
│       │                  │              │                                │
│       │             ┌────┴────┐    Tier 1: ccall / llvmcall             │
│       │             │ Tier    │    Tier 2: JITManager.invoke()           │
│       │             │ Select  │         or AOT thunk ccall              │
│       │             └────┬────┘                                         │
│       │                  │                                              │
│       ▼                  ▼                                              │
│  JLCSIRGenerator.jl ──→ MLIRNative.jl ──→ JITManager.jl                │
│  (ir_gen/ submodules)    (libJLCS.so)      (lock-free cache)            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline

The full lifecycle of a C/C++ project through RepliBuild proceeds in six stages:

```
C/C++ Source + [dependencies] (git / local / system)
    │
    ▼
┌─ 1. Discovery ──────────────────────────────────────────────┐
│  Scan source files, parse #include graph, resolve deps,     │
│  generate replibuild.toml, register in global registry      │
└──────────────────────────────────┬──────────────────────────┘
                                   │
    ▼                              │
┌─ 2. Dependency Resolution ───────┘──────────────────────────┐
│  Clone/update git repos, filter excludes, inject sources    │
│  into compile graph. (DependencyResolver.jl)                │
└──────────────────────────────────┬──────────────────────────┘
                                   │
    ▼                              │
┌─ 3. Compilation ─────────────────┘──────────────────────────┐
│  Clang → per-file LLVM IR (.ll)                             │
│  Incremental mtime cache, parallel dispatch                 │
│  Template instantiation (forced via config)                 │
└──────────────────────────────────┬──────────────────────────┘
                                   │
    ▼                              │
┌─ 4. Linking ─────────────────────┘──────────────────────────┐
│  IR merge → LTO optimization → shared library (.so/.dylib)  │
│  Optional: _lto.ll (for Base.llvmcall embedding)            │
│  Optional: _thunks.so (pre-compiled MLIR AOT thunks)        │
└──────────────────────────────────┬──────────────────────────┘
                                   │
    ▼                              │
┌─ 5. Wrapping ────────────────────┘──────────────────────────┐
│  DWARF introspection → raw ccall/llvmcall wrappers          │
│  Idiomatic mutable structs with finalizers                  │
│  Tier selection: ccall vs JIT vs LTO per function           │
└──────────────────────────────────┬──────────────────────────┘
                                   │
    ▼                              │
┌─ 6. JIT Init (on demand) ───────┘──────────────────────────┐
│  MLIR IR generation → JLCS dialect → LLVM lowering → JIT   │
│  Lock-free symbol cache for hot-path dispatch               │
│  Or: AOT thunks.so loaded at module parse time              │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Map

### Core API

| Module | File | Responsibility |
|--------|------|----------------|
| **RepliBuild** | `src/RepliBuild.jl` | Top-level module. Exports `discover`, `build`, `wrap`, `use`, `check_environment`. |
| **ConfigurationManager** | `src/ConfigurationManager.jl` | Load, validate, merge `replibuild.toml` into a typed `RepliBuildConfig` struct. |
| **LLVMEnvironment** | `src/LLVMEnvironment.jl` | Detect system LLVM/Clang toolchain; fall back to LLVM_full_jll. |
| **EnvironmentDoctor** | `src/EnvironmentDoctor.jl` | `check_environment()` — validates LLVM 21+, Clang, mlir-tblgen, CMake, libJLCS.so. Returns `ToolchainStatus`. |

### Discovery & Dependencies

| Module | File | Responsibility |
|--------|------|----------------|
| **Discovery** | `src/Discovery.jl` | Walk a C++ project directory, resolve `#include` graph, emit `replibuild.toml`. |
| **DependencyResolver** | `src/DependencyResolver.jl` | Fetch git/local/system deps declared in `[dependencies]`, filter excludes, inject into compile graph. |
| **PackageRegistry** | `src/PackageRegistry.jl` | Global `~/.replibuild/registry/` — `use()`, `register()`, `list_registry()`, `unregister()`. Artifact caching. |

### Compilation & Linking

| Module | File | Responsibility |
|--------|------|----------------|
| **Compiler** | `src/Compiler.jl` | Per-file C++ → LLVM IR compilation. Incremental mtime cache, parallel dispatch, template instantiation. |
| **BuildBridge** | `src/BuildBridge.jl` | Shell out to `clang`, `llvm-link`, `llvm-opt`, `nm`. Low-level compiler driver. |

### DWARF Extraction

| Module | File | Responsibility |
|--------|------|----------------|
| **DWARFParser** | `src/DWARFParser.jl` | Parse `llvm-dwarfdump` output. Extract `ClassInfo`, `VtableInfo`, `MemberInfo`, `VirtualMethod` structs. Handles unions, bitfields, globals, typedefs. |
| **ASTWalker** | `src/ASTWalker.jl` | Clang.jl-based AST walker for enum extraction (replaces regex-based approach). |
| **ClangJLBridge** | `src/ClangJLBridge.jl` | Clang.jl integration for header parsing. |

### Wrapper Generation

| Module | File | Responsibility |
|--------|------|----------------|
| **Wrapper** | `src/Wrapper.jl` | The largest module. Generates the complete Julia wrapper module: struct definitions, enum mappings, ccall wrappers, llvmcall paths, JIT thunks, idiomatic `mutable struct` types with finalizers, method dispatch proxies. Contains the tier selection logic (`is_ccall_safe()`). |
| **STLWrappers** | `src/STLWrappers.jl` | STL container type detection and accessor generation. |

### MLIR & JIT

| Module | File | Responsibility |
|--------|------|----------------|
| **JLCSIRGenerator** | `src/JLCSIRGenerator.jl` | Emit MLIR JLCS dialect IR from `VtableInfo`. Orchestrates submodules in `src/ir_gen/`. |
| **ir_gen/TypeUtils** | `src/ir_gen/TypeUtils.jl` | C++ → MLIR type mapping (`double`→`f64`, `int*`→`!llvm.ptr`, etc.). |
| **ir_gen/StructGen** | `src/ir_gen/StructGen.jl` | Generate `jlcs.type_info` operations from `ClassInfo`. Topological sort for inheritance. |
| **ir_gen/FunctionGen** | `src/ir_gen/FunctionGen.jl` | Generate `func.func` thunks for virtual methods and regular functions. |
| **ir_gen/STLContainerGen** | `src/ir_gen/STLContainerGen.jl` | Generate MLIR thunks for STL container accessors. |
| **MLIRNative** | `src/MLIRNative.jl` | Low-level `ccall` bindings to `libJLCS.so`: context management, module parsing, JIT engine, `lower_to_llvm`, `lookup`. |
| **JITManager** | `src/JITManager.jl` | Singleton `GLOBAL_JIT`. Lock-free symbol cache, arity-specialized `invoke` (0–4 args), `@generated` ABI dispatch for scalar vs struct returns. |

### JLCS MLIR Dialect (C++)

| File | Role |
|------|------|
| `src/mlir/JLCSDialect.td` | Dialect registration and namespace definition. |
| `src/mlir/JLCSOps.td` | Operation definitions: `type_info`, `get_field`, `set_field`, `vcall`, `load_array_element`, `store_array_element`, `ffe_call`. |
| `src/mlir/Types.td` | Type definitions: `!jlcs.c_struct<>`, `!jlcs.array_view<>`. |
| `src/mlir/JLCS.td` | Aggregate include for TableGen. |
| `src/mlir/JLInterfaces.td` | Interface definitions for the dialect. |
| `src/mlir/CMakeLists.txt` | Build config: TableGen processing, whole-archive JIT linking. |
| `src/mlir/build.sh` | Build script. Produces `src/mlir/build/libJLCS.so`. |
| `src/mlir/impl/` | C++ implementation files for dialect operations. |

### Introspection Toolkit

| Module | File | Responsibility |
|--------|------|----------------|
| **Introspect** | `src/Introspect.jl` | Umbrella module for binary analysis and diagnostics. |
| **Binary** | `src/Introspect/Binary.jl` | `symbols()`, `dwarf_info()`, `disassemble()` — binary analysis. |
| **Julia** | `src/Introspect/Julia.jl` | Julia IR introspection utilities. |
| **LLVM** | `src/Introspect/LLVM.jl` | LLVM pass tooling and IR inspection. |
| **Benchmarking** | `src/Introspect/Benchmarking.jl` | `benchmark()` with configurable samples. |
| **DataExport** | `src/Introspect/DataExport.jl` | Export results to JSON. |
| **Types** | `src/Introspect/Types.jl` | Shared types for the introspection subsystem. |

---

## Two-Tier Dispatch Model

The core architectural decision: every function is analyzed and routed to one of two calling tiers based on ABI complexity.

### Tier 1 — Direct `ccall` (and optional LTO `llvmcall`)

For functions with simple, POD-safe signatures. Zero overhead.

**Conditions (all must hold):**
- No STL container types in parameters or return
- Return type is: primitive | pointer | void | small aligned struct (≤16 bytes)
- All parameters are: primitive | pointer | small struct — NOT unions, NOT packed structs, NOT non-POD classes

When `enable_lto = true`, eligible Tier 1 functions are upgraded to `Base.llvmcall`. The C++ LLVM IR is embedded as a module constant and passed directly to Julia's JIT compiler, allowing **cross-language inlining** — Julia can inline C++ code directly into hot loops.

```julia
# LTO path — Julia's JIT sees the C++ IR and can inline it
function add(a::Cint, b::Cint)::Cint
    if !isempty(LTO_IR)
        return Base.llvmcall((LTO_IR, "_Z3addii"), Cint, Tuple{Cint, Cint}, a, b)
    else
        return ccall((:_Z3addii, LIBRARY_PATH), Cint, (Cint, Cint), a, b)
    end
end
```

**LTO eligibility** (additional constraints beyond Tier 1):
- NOT a virtual method
- Does NOT return a struct by value
- No `Cstring` parameters or return (llvmcall doesn't auto-convert)

### Tier 2 — MLIR JIT (or AOT Thunks)

For functions requiring complex ABI marshalling: packed structs, unions, large struct returns, C++ virtual dispatch.

**Two sub-modes:**

| Mode | Config | Mechanism | Startup | Call Overhead |
|------|--------|-----------|---------|---------------|
| **JIT** | `aot_thunks = false` | `JITManager.invoke()` at runtime | First-call JIT cost | Lock-free after first call |
| **AOT** | `aot_thunks = true` | Pre-compiled `_thunks.so` + `ccall` | Zero (pre-compiled) | Same as ccall |

### Tier Selection Flow

```
                    ┌──────────────┐
                    │  Function    │
                    │  Signature   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ is_ccall_    │
                    │ safe()?      │
                    └──┬───────┬───┘
                   yes │       │ no
                       │       │
              ┌────────▼──┐  ┌─▼──────────────┐
              │  Tier 1    │  │ enable_lto &&   │
              │  ccall     │  │ !virtual &&     │
              └──────┬─────┘  │ !struct_ret &&  │
                     │        │ !Cstring?       │
              ┌──────▼──┐     └──┬──────────┬───┘
              │ LTO?    │     yes │          │ no
              └──┬───┬──┘        │          │
              yes│   │no   ┌─────▼─────┐ ┌──▼───────────┐
                 │   │     │ Tier 2    │ │ aot_thunks?  │
          ┌──────▼┐  │     │ llvmcall  │ └──┬────────┬──┘
          │llvm   │  │     │ (LTO AOT) │ yes│        │no
          │call   │  │     └───────────┘    │        │
          └───────┘  │                ┌─────▼───┐ ┌──▼──────────┐
                     │                │ Tier 2  │ │ Tier 2     │
              ┌──────▼──┐             │ ccall   │ │ JITManager │
              │ ccall   │             │ thunks  │ │ .invoke()  │
              │ (std)   │             │ .so     │ │ (runtime)  │
              └─────────┘             └─────────┘ └────────────┘
```

---

## The JLCS MLIR Dialect

JLCS (Julia C-Struct) is a custom MLIR dialect that models C-ABI-compatible struct layout and foreign function execution. It is the core of Tier 2 dispatch.

### Type System

```mlir
// C struct with explicit field types, byte offsets, and packing
!jlcs.c_struct<"MyStruct", [i32, i64, f64], [0 : i64, 4 : i64, 12 : i64], packed = false>

// Strided multi-dimensional array descriptor (Julia, NumPy, C++, Rust compatible)
!jlcs.array_view<f64, 3>    // 3D array of float64
```

The `c_struct` type carries the full ABI contract: field types, byte offsets, and packing flag. This ensures correct marshalling regardless of platform alignment rules.

The `array_view` type maps to a runtime struct: `{data_ptr, dims_ptr, strides_ptr, rank}` — enabling zero-copy interop with Julia arrays, NumPy ndarrays, and C++ containers.

### Operations

| Operation | Signature | Semantics |
|-----------|-----------|-----------|
| `jlcs.type_info` | `(typeName, structType, superType)` | Register a struct type and its C++ inheritance mapping. Module-level declaration. |
| `jlcs.get_field` | `(struct, fieldOffset) → value` | Read a field at a byte offset from a C struct pointer. |
| `jlcs.set_field` | `(struct, value, fieldOffset)` | Write a field at a byte offset into a C struct pointer. |
| `jlcs.vcall` | `(class, args..., vtable_offset, slot) → result` | Virtual method dispatch: read vtable pointer → index slot → call function pointer. |
| `jlcs.load_array_element` | `(view, indices...) → element` | Strided array read: `offset = Σ(index_i × stride_i)`. |
| `jlcs.store_array_element` | `(value, view, indices...)` | Strided array write. |
| `jlcs.ffe_call` | `(args...) → results...` | Call an external C function using FFE metadata. |

### Generated MLIR Module Structure

For each C++ class with virtual methods, the IR generator emits:

```mlir
module {
  // 1. External dispatch declarations (one per method symbol)
  llvm.func @_ZN4Base3fooEv(!llvm.ptr) -> i32

  // 2. Type info (one per class — registers struct layout)
  jlcs.type_info "Base",
    !jlcs.c_struct<"Base", [!llvm.ptr, i32, i32],
                   [0 : i64, 8 : i64, 12 : i64], packed = false>, ""

  // 3. Thunk wrappers (one per virtual method — bridges Julia → C++)
  func.func @thunk__ZN4Base3fooEv(%arg0: !llvm.ptr) -> i32 {
    %result = llvm.call @_ZN4Base3fooEv(%arg0) : (!llvm.ptr) -> i32
    return %result : i32
  }
}
```

This module is lowered through MLIR's standard pipeline: `func` → `llvm` dialect → native LLVM IR → machine code via the JIT execution engine.

### Build

The dialect is built via CMake with TableGen:

```
src/mlir/
├── *.td              → TableGen definitions
├── impl/             → C++ operation implementations
├── CMakeLists.txt    → Build config (TableGen, whole-archive JIT linking)
├── build.sh          → Build script
└── build/
    └── libJLCS.so    → Output (required for JIT tier)
```

Dependencies: `MLIRExecutionEngine`, `MLIRTargetLLVMIRExport`, `MLIRLLVMToLLVMIRTranslation`.

---

## DWARF Extraction

The DWARF parser is the bridge between compiled C++ binaries and Julia wrapper generation. It extracts everything needed for ABI-correct bindings from debug metadata.

### Data Structures

```julia
struct VirtualMethod
    name::String              # "foo"
    mangled_name::String      # "_ZN4Base3fooEv"
    slot::Int                 # Vtable slot index
    return_type::String       # C++ type name
    parameters::Vector{String}
end

struct MemberInfo
    name::String              # Field name
    type_name::String         # C++ type (int, double*, std::vector<>, ...)
    offset::Int               # Byte offset in struct
end

struct ClassInfo
    name::String
    vtable_ptr_offset::Int    # Usually 0
    base_classes::Vector{String}
    virtual_methods::Vector{VirtualMethod}
    members::Vector{MemberInfo}
    size::Int                 # Total struct size in bytes
end

struct VtableInfo
    classes::Dict{String, ClassInfo}
    vtable_addresses::Dict{String, UInt64}
    method_addresses::Dict{String, UInt64}
end
```

### Extraction Flow

```
llvm-dwarfdump binary.so
    │
    ├─ DW_TAG_class_type / DW_TAG_structure_type
    │      ├─ DW_AT_name, DW_AT_byte_size
    │      ├─ DW_TAG_member → MemberInfo (name, type, DW_AT_data_member_location)
    │      ├─ DW_TAG_subprogram [virtual] → VirtualMethod (name, mangled, slot)
    │      └─ DW_TAG_inheritance → base_classes
    │
    ├─ DW_TAG_enumeration_type → Enum definitions
    ├─ DW_TAG_union_type → Union layout
    ├─ DW_TAG_variable → Global variables
    └─ DW_TAG_typedef → Type aliases
```

Beyond classes, the parser also extracts unions, bitfields, global variables, typedefs, and varargs markers — feeding all of this into the wrapper generator.

---

## IR Generation (DWARF → MLIR)

The `JLCSIRGenerator` transforms parsed DWARF metadata into MLIR source text, which is then parsed and JIT-compiled by `MLIRNative`.

### Type Mapping

| C++ Type | MLIR Type |
|----------|-----------|
| `double` | `f64` |
| `float` | `f32` |
| `int` / `unsigned int` | `i32` |
| `long` / `long long` | `i64` |
| `char` / `int8_t` | `i8` |
| `void` | `none` |
| `T*`, `T&` | `!llvm.ptr` |
| `std::vector<T>` | `!llvm.ptr` (opaque) |
| Unknown | `!llvm.ptr` (fallback) |

### Generation Submodules

| Module | Input | Output |
|--------|-------|--------|
| `TypeUtils.jl` | C++ type string | MLIR type string |
| `StructGen.jl` | `ClassInfo` + members | `jlcs.type_info` operation |
| `FunctionGen.jl` | `VirtualMethod` | `func.func @thunk_...` wrapper |
| `STLContainerGen.jl` | STL method metadata | Accessor thunks for `size()`, `data()`, etc. |

---

## JIT Manager — Lock-Free Dispatch

The JIT manager provides the runtime execution path for Tier 2 functions.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  GLOBAL_JIT (singleton)          │
│                                                   │
│  mlir_ctx        → Ptr{Cvoid}  (MLIR context)    │
│  jit_engine      → Ptr{Cvoid}  (execution engine) │
│  compiled_symbols → Dict{String, Ptr{Cvoid}}      │
│  vtable_info     → VtableInfo                     │
│  lock            → ReentrantLock                  │
└─────────────────────────────────────────────────┘
```

### Lock-Free Lookup (Double-Check Pattern)

```
invoke("_mlir_ciface_foo_thunk", RetType, args...)
    │
    ▼
_lookup_cached(func_name)
    │
    ├─ FAST PATH: Dict read (no lock) ──→ cache hit → return Ptr
    │
    └─ SLOW PATH: lock → double-check → MLIRNative.lookup() → cache → return Ptr
```

- **Hot path**: Single `Dict` read with no synchronization. Julia's `Dict` is safe for concurrent reads under a single-writer pattern.
- **Cold path**: Lock acquisition, JIT symbol resolution, cache insertion. Only happens once per symbol.

### Calling Convention

All Tier 2 functions use a unified calling convention for MLIR `ciface` thunks:

```
Arguments:  All passed as Ref{T} → Ptr{Cvoid}[] (pointer-to-value array)
            inner_ptrs = [ptr_to_arg1, ptr_to_arg2, ..., ptr_to_argN]

Scalar return:   T    ciface(void** args_ptr)
Struct return:   void ciface(T* sret_buf, void** args_ptr)
Void return:     void ciface(void** args_ptr)
```

### Arity Specialization

To avoid heap-allocating `Any[]` for small argument counts, the JIT manager provides hand-specialized `invoke` methods for 0–4 arguments. Each creates stack-allocated `Ref`s and a fixed-size `Ptr{Cvoid}[]`, avoiding all boxing:

```julia
function invoke(func_name::String, ::Type{T}, a1, a2) where T
    fptr = _lookup_cached(func_name)
    r1 = Ref(a1); r2 = Ref(a2)
    inner_ptrs = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, r1),
        Base.unsafe_convert(Ptr{Cvoid}, r2)
    ]
    GC.@preserve r1 r2 begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end
```

A variadic fallback handles 5+ arguments with dynamic allocation.

Return dispatch is resolved at compile time via `@generated`:
- `isprimitivetype(T)` → direct `ccall` return
- Otherwise → `sret` buffer allocation, `ccall` with out-pointer, dereference

---

## Wrapper Generation

`Wrapper.jl` is the largest module and the heart of the code generator. It takes DWARF metadata + compiled artifacts and emits a complete, loadable Julia module.

### What Gets Generated

```
<project>/julia/
├── <LibName>.so                   # Compiled shared library
├── compilation_metadata.json      # Symbol + DWARF metadata
└── <ModuleName>.jl                # Generated Julia module containing:
    │
    ├── Module-level constants
    │   ├── LIBRARY_PATH           # Path to .so
    │   ├── LTO_IR                 # Embedded LLVM IR (if enable_lto)
    │   └── THUNKS_LTO_IR          # Embedded thunk IR (if aot_thunks)
    │
    ├── Struct definitions
    │   ├── Correct field order, alignment padding (_pad_N fields)
    │   ├── Forward declarations for circular references
    │   ├── Base class member flattening
    │   └── NTuple{N,UInt8} backing for unions
    │
    ├── Enum definitions (@enum with correct underlying types)
    │
    ├── Function wrappers
    │   ├── Tier 1: ccall / llvmcall (with LTO fallback)
    │   ├── Tier 2: JITManager.invoke() or AOT thunk ccall
    │   ├── Variadic overloads (from [wrap.varargs] config)
    │   └── Global variable accessors (cglobal + unsafe_load)
    │
    ├── Idiomatic wrappers
    │   ├── mutable struct types (factory + destructor clustering)
    │   ├── GC-managed finalizers (ManagedX types)
    │   ├── Multiple-dispatch method proxies
    │   └── Base.unsafe_convert for pointer passing
    │
    └── Bitfield accessors (bit-shift extraction)
```

### Idiomatic Wrapper Strategy

RepliBuild doesn't just emit raw `ccall` bindings. It clusters related C++ functions by class name:

1. **Factory detection**: `create_circle()` → constructor for `Circle`
2. **Destructor detection**: `delete_shape()` / `~Shape()` → finalizer
3. **Method clustering**: `area(Shape*)` → `area(s::ManagedShape)`

The result is `mutable struct ManagedShape` with:
- A raw `Ptr{Cvoid}` handle
- A registered `finalizer` calling the C++ destructor
- Multiple-dispatch method proxies that pass the pointer via `Base.unsafe_convert`

---

## Configuration (`replibuild.toml`)

Generated by `discover()`, hand-editable. Key sections:

```toml
[project]
name = "MyProject"
uuid = "auto-generated"

[compile]
flags = ["-std=c++17", "-fPIC", "-O3"]
source_files = ["src/*.cpp"]       # Auto-detected
include_dirs = ["include/"]
parallel = true
aot_thunks = false                 # Pre-compile MLIR thunks → _thunks.so

[link]
enable_lto = false                 # Emit _lto.ll for Base.llvmcall paths
optimization_level = "3"

[binary]
type = "shared"                    # shared | static
strip_symbols = false

[wrap]
style = "clang"
use_clang_jl = true

[wrap.varargs]                     # Typed overloads for variadic functions
printf = [["Cstring", "Cint"], ["Cstring", "Cdouble"]]

[types]
strictness = "warn"                # strict | warn | permissive
allow_unknown_structs = true
allow_function_pointers = true
templates = ["std::vector<int>"]   # Force Clang to emit DWARF for these
template_headers = ["<vector>"]

[cache]
enabled = true
directory = ".replibuild_cache"

[dependencies.cjson]               # External git dependency
type = "git"
url = "https://github.com/DaveGamble/cJSON"
tag = "v1.7.18"
exclude = ["test", "fuzzing"]
```

---

## Caching Strategy

RepliBuild uses two levels of caching:

### 1. Per-File IR Cache (mtime-based)

Each source file's compiled LLVM IR is cached in `.replibuild_cache/` keyed by filepath. On recompilation, only files with changed `mtime` are recompiled.

### 2. Project-Level Content Hash (SHA256)

A hash of `replibuild.toml` + all source contents + all header contents + git HEAD. If the hash matches the cached artifacts, `build()` returns in sub-second time without invoking any compiler.

### 3. Global Registry Cache

`~/.replibuild/builds/<hash>/` stores full build artifacts. The `use()` function checks this cache first, enabling instant loads of previously built packages.

### 4. Toolchain Cache

`~/.replibuild/toolchain.toml` caches the result of environment probing (LLVM/Clang detection) with a 24-hour TTL.

---

## Performance Characteristics

### Per-Call Overhead vs Bare `ccall`

| Scenario | Tier | Median | vs Bare ccall |
|----------|------|--------|---------------|
| `scalar_add` | Pure Julia | 30 ns | 1.0× |
| `scalar_add` | Bare `ccall` | 30 ns | 1.0× (baseline) |
| `scalar_add` | Wrapper `ccall` | 40 ns | 1.33× |
| `scalar_add` | **LTO `llvmcall`** | **30 ns** | **1.0×** |
| `pack_record` | Bare `ccall` (unsafe) | ⚠ crashes | — |
| `pack_record` | Wrapper `ccall` (DWARF) | 80 ns | safe |

### Hot Loop (1M iterations, `add_to(acc, val)`)

| Tier | ns/iter | Note |
|------|---------|------|
| Pure Julia | 0.677 | `@inbounds` native loop |
| Bare `ccall` | 1.800 | Hand-written FFI |
| Wrapper `ccall` | 2.026 | Generated (LTO disabled) |
| **LTO `llvmcall`** | **0.677** | **Julia JIT inlines C++ IR** |
| Whole loop in C++ | 0.997 | Single `ccall` to C++ loop |

The LTO path matches pure Julia performance by allowing the JIT to see and optimize across the FFI boundary.

---

## Test Suite

Self-contained tests with their own `replibuild.toml` configs:

| Test | Target | What It Validates |
|------|--------|-------------------|
| **stress_test** | Custom C++ | DWARF extraction, struct layout, type introspection |
| **vtable_test** | C++ inheritance | Virtual dispatch via MLIR JIT (Circle/Rectangle polymorphism) |
| **callback_test** | Bidirectional FFI | Julia `@cfunction` passed to C++ event loops |
| **benchmark_test** | Matrix views | Zero-copy struct pointers, ~94ns 4×4 matmul |
| **jit_edge_test** | Scalars/structs | 3-tier benchmark: ccall vs wrapper vs MLIR JIT |
| **lto_benchmark_test** | LTO paths | `llvmcall` inlining verification |
| **Lua 5.4.7** | Real-world C | State management, stack ops, code eval, coroutines |
| **Duktape 2.7.0** | 101K-line C | Monolithic amalgamation compile, JS evaluation |
| **SQLite 3.49.1** | 261K-line C | Full C API wrap with varargs |
| **Eigen** | C++ template lib | Template instantiation, namespace handling |

External sources (Lua, Duktape, SQLite) are downloaded on demand via `setup.jl` scripts — not vendored.

```bash
# Run main test suite
julia --project=. test/runtests.jl

# Run comprehensive integration suite
julia --project=. test/run_comprehensive.jl

# Run individual test
julia --project=. test/test_mlir.jl
```

---

## Generated Output Layout

```
<project>/
├── replibuild.toml                    # Configuration (generated, editable)
├── build/                             # LLVM IR files (.ll), intermediates
├── julia/
│   ├── <LibName>.so                   # Compiled shared library
│   ├── <LibName>_lto.ll               # LTO IR (if enable_lto)
│   ├── <LibName>_thunks.so            # AOT thunks (if aot_thunks)
│   ├── compilation_metadata.json      # Symbol + DWARF metadata
│   └── <ModuleName>.jl                # Generated Julia wrapper module
└── .replibuild_cache/                 # Incremental compile cache
```

---

## Key Design Decisions

**Source-based, not binary-based.** RepliBuild deliberately bypasses JLLs and BinaryBuilder. By owning the compilation, it gets perfect DWARF metadata, can enable LTO across the FFI boundary, and can tailor binaries to the host machine. The tradeoff is requiring a local LLVM 21+ toolchain.

**DWARF as the source of truth.** Rather than parsing headers (which miss ABI details like padding, vtable layout, and actual sizes), RepliBuild compiles with `-g` and reads the debug metadata that the compiler itself emitted. This means struct layout, inheritance, and virtual dispatch tables are always correct.

**Custom MLIR dialect over ad-hoc codegen.** The JLCS dialect provides a principled intermediate representation for C++ interop. Operations like `vcall` and `get_field` encode ABI semantics that would be error-prone to emit as raw LLVM IR directly. The MLIR framework handles lowering, optimization, and JIT compilation.

**Lock-free hot path.** The JIT manager's double-check caching pattern means that after the first call to any JIT function, subsequent calls are a single unsynchronized `Dict` lookup — no locks, no allocation, no JIT overhead.

**Graceful degradation.** If `libJLCS.so` isn't built, Tier 1 (ccall) still works. If LTO is disabled, wrappers fall back to standard `ccall`. If the toolchain is incomplete, `check_environment()` tells the user exactly what's missing with OS-specific install instructions.
