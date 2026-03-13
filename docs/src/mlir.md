# MLIR & JLCS Dialect

## Background: what is MLIR?

[MLIR](https://mlir.llvm.org/) (Multi-Level Intermediate Representation) is a compiler infrastructure developed as part of the LLVM project. Unlike traditional compilers that operate on a single IR (e.g., LLVM IR), MLIR supports **multiple levels of abstraction** through user-defined *dialects* — each dialect defines its own types, operations, and semantics. Dialects can be progressively *lowered* from high-level domain-specific operations down to LLVM IR and then to native machine code.

MLIR is used in production by TensorFlow (MHLO dialect), PyTorch (Torch-MLIR), and hardware compilers (CIRCT). RepliBuild uses MLIR because C++ ABI interop involves operations (struct field access at byte offsets, vtable-based virtual dispatch, strided array views) that are error-prone to express directly as LLVM IR but natural to represent as structured, typed MLIR operations.

**Reference:** [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/), [Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)

## Why a custom dialect?

Calling a C++ virtual method from Julia requires:

1. Reading the vtable pointer from the object at a known byte offset
2. Indexing into the vtable to get the function pointer for the correct slot
3. Calling that function pointer with the correct calling convention (sret for struct returns, pointer-to-value for arguments)

Encoding this as raw LLVM IR is possible but fragile — byte offsets must be manually computed, pointer casts must be correct, and struct return conventions vary by platform. A single mistake produces silent memory corruption.

The JLCS dialect expresses these operations as **typed, verifiable IR** that the MLIR framework can validate, optimize, and lower to correct LLVM IR automatically. The dialect also carries ABI metadata (field offsets, packing flags, struct sizes) that would be lost if emitted directly as LLVM IR.

## JLCS dialect specification

**JLCS** (Julia C-Struct) is a custom MLIR dialect that models C-ABI-compatible struct layout and foreign function execution. It is the core of [Tier 2 dispatch](@ref "Three-tier dispatch").

**Source files:**

| File | Role |
|------|------|
| `src/mlir/JLCSDialect.td` | Dialect registration and namespace (`jlcs`) |
| `src/mlir/JLCSOps.td` | Operation definitions |
| `src/mlir/Types.td` | Type definitions |
| `src/mlir/JLInterfaces.td` | Interface definitions |
| `src/mlir/impl/` | C++ implementations for operation verification and lowering |

### Type system

The JLCS dialect defines two custom types.

#### `!jlcs.c_struct` — C-ABI-compatible struct

**Defined in:** `src/mlir/Types.td`

Models a C struct with explicit field types, byte offsets, and a packing flag. This type carries the full ABI contract — the MLIR lowering uses these offsets to generate correct `getelementptr` instructions regardless of platform alignment rules.

**TableGen definition:**

```
def CStructType : JLCS_Type<"CStruct", "c_struct"> {
  let parameters = (ins
    "StringAttr":$juliaTypeName,
    ArrayRefParameter<"Type", "field types">:$fieldTypes,
    "ArrayAttr":$fieldOffsets,
    "bool":$isPacked
  );
}
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `juliaTypeName` | `StringAttr` | Julia-side type name (e.g., `"MyModule.Outer"`) |
| `fieldTypes` | `Type[]` | Ordered list of MLIR types for each field |
| `fieldOffsets` | `ArrayAttr` of `i64` | Byte offset of each field from struct base |
| `isPacked` | `bool` | Whether the struct uses `__attribute__((packed))` layout |

**MLIR syntax:**

```mlir
!jlcs.c_struct<"MyStruct", [i32, i64, f64], [0 : i64, 4 : i64, 12 : i64], packed = false>
```

This declares a struct `MyStruct` with three fields: an `i32` at byte offset 0, an `i64` at offset 4, and an `f64` at offset 12. The `packed = false` flag indicates standard alignment rules apply.

#### `!jlcs.array_view` — strided multi-dimensional array descriptor

**Defined in:** `src/mlir/Types.td`

A universal array descriptor for zero-copy interop with Julia arrays, NumPy ndarrays, and C++ containers. The rank (number of dimensions) is a compile-time constant; the actual dimensions and strides are runtime values.

**TableGen definition:**

```
def ArrayViewType : JLCS_Type<"ArrayView", "array_view"> {
  let parameters = (ins
    "Type":$elementType,
    "unsigned":$rank
  );
}
```

**Runtime memory layout:**

```c
struct ArrayView {
    T*       data_ptr;     // offset 0:  pointer to element data
    int64_t* dims_ptr;     // offset 8:  pointer to dimension sizes
    int64_t* strides_ptr;  // offset 16: pointer to stride values (in elements)
    int64_t  rank;         // offset 24: number of dimensions
};
```

**MLIR syntax:**

```mlir
!jlcs.array_view<f64, 3>    // 3D array of float64
```

This layout is compatible with Julia's `Array` (column-major strides), NumPy's `ndarray` (arbitrary strides), and C++ row-major arrays, enabling zero-copy data sharing across language boundaries.

### Operations

The JLCS dialect defines seven operations, all specified in `src/mlir/JLCSOps.td`.

#### `jlcs.type_info` — register struct type and layout

Declares a `CStruct` type and its C++ base class mapping. Placed in the module's top-level region as a module-scope declaration.

```mlir
jlcs.type_info "Base",
    !jlcs.c_struct<"Base", [!llvm.ptr, i32, i32],
                   [0 : i64, 8 : i64, 12 : i64], packed = false>, ""
```

| Argument | Type | Description |
|----------|------|-------------|
| `typeName` | `StrAttr` | Julia-side type name |
| `structType` | `TypeAttr` | Must be a `CStructType` |
| `superType` | `StrAttr` | Base class name (empty string if none) |

The `superType` field enables the MLIR lowering to handle C++ inheritance chains — base class members are flattened into the derived struct at their correct offsets.

#### `jlcs.get_field` — read a struct field

Read a field at a byte offset from a C struct pointer.

```mlir
%value = jlcs.get_field %struct_ref { fieldOffset = 4 : i64 } : (!llvm.ptr) -> i32
```

Lowers to a `getelementptr` + `load` sequence with the correct byte offset. The field type is carried in the operation's result type, ensuring type safety through the lowering pipeline.

#### `jlcs.set_field` — write a struct field

Write a value at a byte offset into a C struct pointer.

```mlir
jlcs.set_field %struct_ref, %new_value { fieldOffset = 4 : i64 } : (!llvm.ptr, i32) -> ()
```

Lowers to a `getelementptr` + `store` sequence.

#### `jlcs.vcall` — virtual method dispatch

Call a C++ virtual method via vtable lookup. This is the operation that makes Tier 2 dispatch possible for polymorphic C++ classes.

```mlir
%result = jlcs.vcall @Base::foo(%obj) {vtable_offset = 0 : i64, slot = 0 : i64}
    : (!llvm.ptr) -> i32
```

| Argument | Type | Description |
|----------|------|-------------|
| `class_name` | `SymbolRefAttr` | Class name for the vtable |
| `args` | `Variadic<AnyType>` | Arguments (first is always the object pointer) |
| `vtable_offset` | `I64Attr` | Byte offset of the vptr within the object (usually 0) |
| `slot` | `I64Attr` | Index into the vtable function pointer array |

**Lowering semantics:**

1. Load vtable pointer from object at `vtable_offset`
2. Load function pointer from `vtable[slot]`
3. Call the function pointer with the object pointer + remaining arguments

#### `jlcs.load_array_element` — strided array read

Read an element from a multi-dimensional strided array.

```mlir
%elem = jlcs.load_array_element %view[%i, %j, %k] : !jlcs.array_view<f64, 3> -> f64
```

**Index computation:** `linear_offset = sum(index_i * stride_i)` for each dimension. This supports both row-major and column-major layouts depending on the stride values.

#### `jlcs.store_array_element` — strided array write

Write an element to a multi-dimensional strided array.

```mlir
jlcs.store_array_element %value, %view[%i, %j] : f64, !jlcs.array_view<f64, 2>
```

#### `jlcs.ffe_call` — foreign function execution

Call an external C function using FFE (Foreign Function Execution) metadata.

```mlir
%result = jlcs.ffe_call(%arg0, %arg1) : (i32, !llvm.ptr) -> i32
```

This is a general-purpose foreign call operation used for non-virtual C functions that still require MLIR-level ABI handling (e.g., struct return conventions).

## IR generation pipeline

The path from compiled C++ binary to executable MLIR thunks involves three stages.

### Stage 1: DWARF to structured metadata

**Module:** `src/DWARFParser.jl`

`llvm-dwarfdump` is invoked on the compiled binary. The parser extracts `ClassInfo`, `VtableInfo`, and `VirtualMethod` structs from the DWARF tags (`DW_TAG_class_type`, `DW_TAG_subprogram`, `DW_TAG_inheritance`, etc.).

### Stage 2: metadata to MLIR IR text

**Module:** `src/JLCSIRGenerator.jl`, `src/ir_gen/` submodules

The IR generator transforms parsed DWARF metadata into MLIR source text. Each submodule handles a specific concern:

| Submodule | Input | Output |
|-----------|-------|--------|
| `ir_gen/TypeUtils.jl` | C++ type string | MLIR type string |
| `ir_gen/StructGen.jl` | `ClassInfo` + members | `jlcs.type_info` operation |
| `ir_gen/FunctionGen.jl` | `VirtualMethod` | `func.func @thunk_...` wrapper |
| `ir_gen/STLContainerGen.jl` | STL method metadata | Accessor thunks for `size()`, `data()`, etc. |

**Type mapping** (`src/ir_gen/TypeUtils.jl`):

| C++ Type | MLIR Type |
|----------|-----------|
| `double` | `f64` |
| `float` | `f32` |
| `int`, `unsigned int` | `i32` |
| `long`, `long long` | `i64` |
| `char`, `int8_t` | `i8` |
| `void` | `none` |
| `T*`, `T&` | `!llvm.ptr` |
| `std::vector<T>` | `!llvm.ptr` (opaque) |
| Unknown | `!llvm.ptr` (fallback) |

**Complete generated module example:**

For a C++ class `Base` with virtual methods `foo()` and `bar(int)`, the IR generator produces:

```mlir
module {
  // 1. External dispatch declarations (resolved by the JIT linker)
  llvm.func @_ZN4Base3fooEv(!llvm.ptr) -> i32
  llvm.func @_ZN4Base3barEv(!llvm.ptr, i32) -> i32

  // 2. Type info (registers struct layout with the dialect)
  jlcs.type_info "Base",
      !jlcs.c_struct<"Base", [!llvm.ptr, i32, i32],
                     [0 : i64, 8 : i64, 12 : i64], packed = false>, ""

  // 3. Thunk wrappers (bridge Julia calling convention to C++ ABI)
  func.func @thunk__ZN4Base3fooEv(%arg0: !llvm.ptr) -> i32 {
    %result = llvm.call @_ZN4Base3fooEv(%arg0) : (!llvm.ptr) -> i32
    return %result : i32
  }

  func.func @thunk__ZN4Base3barEv(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
    %result = llvm.call @_ZN4Base3barEv(%arg0, %arg1) : (!llvm.ptr, i32) -> i32
    return %result : i32
  }
}
```

The `llvm.func` declarations at the top tell the JIT execution engine to resolve these symbols from the loaded shared library at link time. The `func.func` thunk wrappers provide the MLIR `ciface` entry points that the Julia-side `JITManager.invoke()` calls into.

### Stage 3: MLIR to machine code

**Module:** `src/MLIRNative.jl`

The generated MLIR text is:

1. **Parsed** into an MLIR module via `MLIRNative.parse_module()`
2. **Lowered** through the MLIR pass pipeline: `jlcs` dialect → `func` dialect → `llvm` dialect → LLVM IR
3. **JIT-compiled** to native machine code by `MLIRExecutionEngine`
4. **Symbol-resolved**: External symbols (`llvm.func` declarations) are linked against the loaded shared library

The `lower_to_llvm()` function in `MLIRNative` drives the full lowering pass pipeline. MLIR dependencies used:

| MLIR Component | Role |
|----------------|------|
| `MLIRExecutionEngine` | JIT compilation and execution |
| `MLIRTargetLLVMIRExport` | MLIR module to LLVM IR translation |
| `MLIRLLVMToLLVMIRTranslation` | LLVM dialect lowering to native LLVM IR |

## JIT manager

**Module:** `src/JITManager.jl`

The JIT manager provides the runtime execution path for Tier 2 functions. It is a singleton (`GLOBAL_JIT`) that manages the MLIR context, JIT execution engine, and compiled symbol cache.

### Architecture

```
+---------------------------------------------------+
|              GLOBAL_JIT (singleton)                |
|                                                    |
|  mlir_ctx        -> Ptr{Cvoid}  (MLIR context)    |
|  jit_engine      -> Ptr{Cvoid}  (execution engine) |
|  compiled_symbols -> Dict{String, Ptr{Cvoid}}      |
|  vtable_info     -> VtableInfo                     |
|  lock            -> ReentrantLock                  |
+---------------------------------------------------+
```

### Lock-free lookup (double-check pattern)

```
invoke("_mlir_ciface_foo_thunk", RetType, args...)
    |
    v
_lookup_cached(func_name)
    |
    +-- FAST PATH: Dict read (no lock) --> cache hit -> return Ptr
    |
    +-- SLOW PATH: lock -> double-check -> MLIRNative.lookup() -> cache -> return Ptr
```

- **Hot path** (cached): Single `Dict` read with no synchronization. Julia's `Dict` is safe for concurrent reads under a single-writer pattern.
- **Cold path** (first call): Lock acquisition, JIT symbol resolution via `MLIRNative.lookup()`, cache insertion. Only happens once per symbol over the lifetime of the process.

### Calling convention

All Tier 2 functions use a unified calling convention for MLIR `ciface` thunks:

| Return type | Signature |
|-------------|-----------|
| Scalar | `T ciface(void** args_ptr)` |
| Struct | `void ciface(T* sret_buf, void** args_ptr)` |
| Void | `void ciface(void** args_ptr)` |

Arguments are passed as **pointers to values** via `Ref{T}` conversion:

```
inner_ptrs = [ptr_to_arg1, ptr_to_arg2, ..., ptr_to_argN]
```

### Arity specialization

To avoid heap-allocating `Any[]` for common small argument counts, the JIT manager provides hand-specialized `invoke` methods for 0 through 4 arguments. Each creates stack-allocated `Ref`s and a fixed-size `Ptr{Cvoid}[]`, avoiding all boxing:

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

Return type dispatch is resolved at compile time via `@generated`:
- `isprimitivetype(T)` → direct `ccall` return
- Otherwise → `sret` buffer allocation, `ccall` with out-pointer, dereference

## Building the dialect

The JLCS MLIR dialect is built as a shared library (`libJLCS.so`) via CMake with TableGen code generation.

**Prerequisites:** LLVM 21+ development headers, CMake 3.20+, `mlir-tblgen`

```bash
cd src/mlir
./build.sh
# Produces: src/mlir/build/libJLCS.so
```

The build configuration (`src/mlir/CMakeLists.txt`) processes the `.td` TableGen definitions to generate C++ header and source files, then links the dialect implementation with whole-archive semantics so the JIT execution engine can discover and register the dialect at runtime.

**Build dependencies:**

| MLIR Library | Role |
|-------------|------|
| `MLIRExecutionEngine` | JIT compilation engine |
| `MLIRTargetLLVMIRExport` | MLIR to LLVM IR export |
| `MLIRLLVMToLLVMIRTranslation` | LLVM dialect to native IR |

`libJLCS.so` is only required for Tier 2 dispatch. If it is not built, Tier 1 (`ccall` / `llvmcall`) still works for all POD-safe functions. Run `RepliBuild.check_environment()` to verify which tiers are available on your system.

## `MLIRNative` API reference

`RepliBuild.MLIRNative` provides the low-level Julia bindings to the MLIR C API.

### Context and modules

```@docs
RepliBuild.MLIRNative.create_context
RepliBuild.MLIRNative.destroy_context
RepliBuild.MLIRNative.@with_context
RepliBuild.MLIRNative.create_module
RepliBuild.MLIRNative.parse_module
RepliBuild.MLIRNative.clone_module
RepliBuild.MLIRNative.print_module
```

### JIT execution

```@docs
RepliBuild.MLIRNative.create_jit
RepliBuild.MLIRNative.destroy_jit
RepliBuild.MLIRNative.register_symbol
RepliBuild.MLIRNative.lookup
RepliBuild.MLIRNative.jit_invoke
RepliBuild.MLIRNative.invoke_safe
```

### Transformations

```@docs
RepliBuild.MLIRNative.lower_to_llvm
```

### Diagnostics

```@docs
RepliBuild.MLIRNative.test_dialect
```
