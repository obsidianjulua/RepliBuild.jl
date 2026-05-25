# MLIR & JLCS Dialect

## For Julia developers: why this page matters

Julia doesn't ship DWARF tools, an IR sanitizer, or a way to call `llvm-as` from a package. RepliBuild fills that gap — and this page documents the piece that handles the cases `ccall` can't: packed structs, virtual method dispatch, strided array views, and unions. If your wrapped function uses any of those, its generated code goes through the MLIR pipeline described here (Tier 2). You don't need to understand MLIR to *use* RepliBuild — tier selection is automatic — but this page explains what happens under the hood when `ccall` isn't safe.

## Background: what is MLIR?

[MLIR](https://mlir.llvm.org/) (Multi-Level Intermediate Representation) is a compiler infrastructure developed as part of the LLVM project. Unlike traditional compilers that operate on a single IR (e.g., LLVM IR), MLIR supports **multiple levels of abstraction** through user-defined *dialects* — each dialect defines its own types, operations, and semantics. Dialects can be progressively *lowered* from high-level domain-specific operations down to LLVM IR and then to native machine code.

MLIR is used in production by TensorFlow (MHLO dialect), PyTorch (Torch-MLIR), and hardware compilers (CIRCT). In the Julia ecosystem, Enzyme's Reactant uses MLIR to optimize IR. RepliBuild uses MLIR differently — not for optimization, but for **safe ABI marshalling**. C++ ABI interop involves operations (struct field access at byte offsets, vtable-based virtual dispatch, strided array views) that are error-prone to express directly as LLVM IR but natural to represent as structured, typed MLIR operations.

**Reference:** [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/), [Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)

## Why a custom dialect?

When RepliBuild's cross-verification detects that a struct's DWARF size doesn't match Julia's alignment calculation (i.e., the struct is packed), or encounters virtual methods or unions, it can't emit a safe `ccall`. These cases need machine code that respects the exact byte offsets from DWARF. That's what JLCS does.

Concretely, calling a C++ virtual method from Julia requires:

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

The JLCS dialect defines fourteen operations, all specified in `src/mlir/JLCSOps.td`. They fall into five groups: metadata, field access, function calls, array access, RAII, and ABI marshalling.

#### `jlcs.type_info` — register struct type and layout

Declares a `CStruct` type, its C++ base class mapping, and its destructor symbol. Placed in the module's top-level region as a module-scope declaration.

```mlir
jlcs.type_info "Base",
    !jlcs.c_struct<"Base", [!llvm.ptr, i32, i32],
                   [0 : i64, 8 : i64, 12 : i64], packed = false>, "", "_ZN4BaseD1Ev"
```

| Argument | Type | Description |
|----------|------|-------------|
| `typeName` | `StrAttr` | Julia-side type name |
| `structType` | `TypeAttr` | Must be a `CStructType` |
| `superType` | `StrAttr` | Base class name (empty string if none) |
| `destructorName` | `StrAttr` | Mangled C++ destructor symbol (empty if none) |

The `superType` field enables the MLIR lowering to handle C++ inheritance chains — base class members are flattened into the derived struct at their correct offsets. The `destructorName` is used by `jlcs.scope` to look up the destructor to invoke at scope exit.

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

Call an external C/C++ function. The operation carries a `callee` symbol reference and a variadic argument list; the lowering pass (`JLCSPasses.cpp`) turns it into an `llvm.call` with platform-correct ABI coercion: `sret` for return-by-value structs above the platform's small-return threshold (16 bytes on x86_64 SysV), `byval` for large struct arguments, and packed struct passing for structs marked packed at the type level. Argument coercion is driven by the MLIR types — the IR generator does not encode ABI rules.

```mlir
%result = jlcs.ffe_call %arg0, %arg1 { callee = @_Z3fooid } : (i32, f64) -> i32
```

This is the call op for any C/C++ function declared `noexcept` or known not to throw. For functions that may throw, see `jlcs.try_call`.

#### `jlcs.try_call` — call with C++ exception catching

Like `jlcs.ffe_call` but lowered to `llvm.invoke` with a landing pad. When a C++ exception escapes the called function, the landing pad catches it, calls `jlcs_set_pending_exception()` with the `what()` message, and returns a zero/null sentinel from the thunk. Julia checks `jlcs_has_pending_exception()` after every Tier 2 call and throws a `CxxException` if set.

```mlir
%result = jlcs.try_call %arg0 { callee = @_Z11might_throwi } : (i32) -> i32
```

The decision between `jlcs.ffe_call` and `jlcs.try_call` is made per function during IR emission, based on the per-function `is_noexcept` flag plus the module-level `may_throw` setting (see [`FunctionGen.jl`](#stage-2-metadata-to-mlir-ir-text)). Functions explicitly marked `noexcept` in C++ skip the landing-pad path entirely, since they cannot throw.

## ABI marshalling operations

These two operations exist specifically because Julia and C/C++ disagree on struct field offsets when packing pragmas are involved. They are emitted by `FunctionGen.jl` whenever the wrapper passes or returns a packed struct and lowered field-by-field by `JLCSPasses.cpp` into `getelementptr`/`load`/`store` and `llvm.extractvalue`/`llvm.insertvalue` sequences.

#### `jlcs.marshal_arg` — aligned-to-packed argument conversion

Reads an aligned Julia struct from a pointer, loads each field at its Julia-aligned byte offset, and assembles them into a packed LLVM struct suitable for passing to a C/C++ function expecting a packed layout. The op carries two attributes describing the layout mismatch:

- `memberTypes` — array of MLIR `TypeAttr` for each struct member, used for typed loads
- `juliaOffsets` — array of `i64` byte offsets for each member in the Julia-aligned layout

```mlir
%packed = jlcs.marshal_arg %julia_ptr
  { memberTypes = [i8, i32, i8], juliaOffsets = [0 : i64, 4 : i64, 8 : i64] }
  : (!llvm.ptr) -> !llvm.struct<packed (i8, i32, i8)>
```

For a `PackedTriplet { char tag; int value; char flag; }` (C layout: 0/1/5, total 6 bytes — Julia layout: 0/4/8, total 12 bytes), this op loads `tag` from `%julia_ptr + 0`, `value` from `%julia_ptr + 4`, and `flag` from `%julia_ptr + 8`, then assembles them into a packed `(i8, i32, i8)` for the C call.

#### `jlcs.marshal_ret` — packed-to-aligned return conversion

The reverse of `marshal_arg`: takes a packed struct value returned from a C/C++ function and repacks it into a Julia-aligned struct. Each member is extracted from the packed return value and inserted into the aligned struct at the corresponding position. The `numMembers` attribute drives the lowering loop.

```mlir
%aligned = jlcs.marshal_ret %packed { numMembers = 3 : i64 }
  : (!llvm.struct<packed (i8, i32, i8)>) -> !llvm.struct<(i8, i32, i8)>
```

The actual byte-level field reconstruction happens during MLIR lowering. From the IR generator's perspective, packing-vs-alignment becomes a single declarative op rather than a fragile sequence of hand-written extractvalue/insertvalue/GEP/load/store instructions.

## RAII operations

These four operations exist for cases where Julia binds a C++ class whose objects must be constructed in place and destructed in reverse-construction order at scope exit. They're emitted by the wrapper generator when a class has explicit constructors and a destructor, and lowered into matched ctor/dtor symbol calls bracketing the scope body.

#### `jlcs.ctor_call` — constructor invocation

Calls a C++ constructor with `this` as the first argument followed by constructor parameters. The first argument must be a pointer to the storage where the object will be constructed (stack alloca or heap allocation).

```mlir
jlcs.ctor_call @_ZN4BaseC1Ei(%ptr, %val) : (!llvm.ptr, i32) -> ()
```

#### `jlcs.dtor_call` — destructor invocation

Calls a C++ destructor with the object pointer as the sole argument. Single-argument semantics — destructors take only `this` and return void.

```mlir
jlcs.dtor_call @_ZN4BaseD1Ev(%ptr) : (!llvm.ptr) -> ()
```

#### `jlcs.scope` — region-based RAII lifetime

Defines a scoped lifetime for a set of C++ objects, with matched destructor symbols. The body region can reference any SSA value visible in the enclosing scope. During lowering, destructor calls are emitted in reverse order at scope exit — matching C++ destruction semantics.

```mlir
%alloca = llvm.alloca 1 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
jlcs.scope(%alloca : !llvm.ptr) dtors([@_ZN4BaseD1Ev]) {
  jlcs.ctor_call @_ZN4BaseC1Ei(%alloca, %val) : (!llvm.ptr, i32) -> ()
  // ... use object ...
  jlcs.yield
}
// destructor automatically emitted here during lowering
```

#### `jlcs.yield` — scope terminator

Terminator for `jlcs.scope` regions. Carries no values; serves only to close the region.

## IR generation pipeline

The path from compiled C++ binary to executable MLIR thunks involves three stages.

### Stage 1: DWARF to structured metadata

**Module:** `src/Builder/DWARFParser.jl`

`llvm-dwarfdump` is invoked on the compiled binary. The parser extracts `ClassInfo`, `VtableInfo`, `VirtualMethod`, and `MemberInfo` structs from the DWARF tags (`DW_TAG_class_type`, `DW_TAG_subprogram`, `DW_TAG_inheritance`, etc.) and pairs the resulting metadata against the `nm` symbol table for linking identity.

### Stage 2: metadata to MLIR IR text

**Module:** `src/IRGen/JLCSIRGenerator.jl` plus the `src/IRGen/ir_gen/` submodules

The IR generator walks the structured DWARF metadata and emits MLIR source text as a string. The emission strategy is deliberately compact — most of the ABI work is deferred to the C++ lowering passes — so the Julia-side codegen stays declarative.

| Submodule | Input | Output |
|-----------|-------|--------|
| `ir_gen/TypeUtils.jl` | C++ type string | MLIR type string |
| `ir_gen/StructGen.jl` | `ClassInfo` + members | `jlcs.type_info` op, plus aligned/packed type strings for use as call signatures |
| `ir_gen/FunctionGen.jl` | function or virtual method metadata | external `func.func private @mangled` decl + public `func.func @mangled_thunk` wrapper |
| `ir_gen/STLContainerGen.jl` | STL method metadata | Accessor thunks for `size()`, `data()`, etc. |

**Type mapping** (`src/IRGen/ir_gen/TypeUtils.jl`):

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

Consider a C++ function returning a packed struct:

```cpp
#pragma pack(push, 1)
typedef struct PackedTriplet { char tag; int value; char flag; } PackedTriplet;
#pragma pack(pop)

PackedTriplet pack_three(char tag, int value, char flag);
```

The IR generator produces (this is the actual output of `FunctionGen.generate_function_thunks`, pre-lowering):

```mlir
module {
  // 1. Type info — registers struct layout with the dialect, including DWARF offsets
  jlcs.type_info "PackedTriplet",
      !jlcs.c_struct<"PackedTriplet", [i8, i32, i8],
                     [0 : i64, 1 : i64, 5 : i64], packed = true>, "", ""

  // 2. External declaration — real C++ symbol, packed layout per DWARF
  func.func private @pack_three(i8, i32, i8) -> !llvm.struct<packed (i8, i32, i8)>

  // 3. Thunk — emit_c_interface causes MLIR to generate _mlir_ciface_pack_three_thunk
  //    with the void** args_ptr calling convention that JITManager.invoke uses
  func.func @pack_three_thunk(%args_ptr: !llvm.ptr)
      -> !llvm.struct<(i8, i32, i8)>
      attributes { llvm.emit_c_interface } {

    // Unpack each argument from the void** args_ptr array
    %idx_1 = arith.constant 0 : i64
    %arg_ptr_1 = llvm.getelementptr %args_ptr[%idx_1]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr_1 = llvm.load %arg_ptr_1 : !llvm.ptr -> !llvm.ptr
    %val_1 = llvm.load %val_ptr_1 : !llvm.ptr -> i8

    %idx_2 = arith.constant 1 : i64
    %arg_ptr_2 = llvm.getelementptr %args_ptr[%idx_2]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr_2 = llvm.load %arg_ptr_2 : !llvm.ptr -> !llvm.ptr
    %val_2 = llvm.load %val_ptr_2 : !llvm.ptr -> i32

    %idx_3 = arith.constant 2 : i64
    %arg_ptr_3 = llvm.getelementptr %args_ptr[%idx_3]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr_3 = llvm.load %arg_ptr_3 : !llvm.ptr -> !llvm.ptr
    %val_3 = llvm.load %val_ptr_3 : !llvm.ptr -> i8

    // Call C++ — returns packed struct (no marshalling needed for scalar args here)
    %ret_packed = jlcs.ffe_call %val_1, %val_2, %val_3
        { callee = @pack_three }
        : (i8, i32, i8) -> !llvm.struct<packed (i8, i32, i8)>

    // Convert packed return into aligned layout for Julia (one declarative op)
    %ret_aligned = jlcs.marshal_ret %ret_packed { numMembers = 3 : i64 }
        : (!llvm.struct<packed (i8, i32, i8)>) -> !llvm.struct<(i8, i32, i8)>

    return %ret_aligned : !llvm.struct<(i8, i32, i8)>
  }
}
```

Three observations about what the IR generator does and doesn't do:

1. **DWARF offsets are baked into the `!jlcs.c_struct` type.** The `[0, 1, 5]` array is read directly from `DW_AT_data_member_location` in the binary's DWARF. The IR generator does not re-derive layout.
2. **ABI work is deferred to the lowering pass.** `jlcs.marshal_ret` is a single op; the field-by-field extractvalue/insertvalue sequence is generated by `JLCSPasses.cpp` during MLIR lowering, not by the Julia codegen.
3. **The `ciface` wrapper is automatic.** The `llvm.emit_c_interface` attribute on the thunk function tells MLIR's LLVM lowering to emit a `_mlir_ciface_pack_three_thunk` entry point with the `void**` argument convention. RepliBuild does not hand-write a trampoline.

If a struct argument were itself packed, the IR generator would emit a `jlcs.marshal_arg` op on the corresponding `%val_*` to perform the aligned→packed conversion before the call. Virtual methods substitute `jlcs.vcall` for `jlcs.ffe_call`. Functions that may throw substitute `jlcs.try_call` for `jlcs.ffe_call`. The thunk skeleton is otherwise identical.

### Stage 3: MLIR to machine code

**Modules:** `src/IRGen/MLIRNative.jl` (Julia side) and `src/mlir/impl/JLCSPasses.cpp` (C++ lowering)

The generated MLIR text is parsed and lowered through a pipeline driven entirely by the C++ side of the dialect:

1. **Parsed** into an MLIR module via `MLIRNative.parse_module()`
2. **Lowered** through the JLCS pass pipeline (defined in `JLCSPasses.cpp`): `jlcs` dialect → `func` dialect → `llvm` dialect → LLVM IR. Each `jlcs.*` op has a corresponding `ConversionPattern` that emits the LLVM-dialect equivalent. The patterns are responsible for ABI coercion: `jlcs.ffe_call` becomes an `llvm.call` with `sret`/`byval` argument coercion when needed; `jlcs.try_call` becomes an `llvm.invoke` with a landing pad; `jlcs.marshal_arg`/`marshal_ret` expand into field-by-field GEP/load/store sequences using the offsets carried in the op attributes; `jlcs.vcall` becomes a vtable load + slot index + indirect call.
3. **JIT-compiled** to native machine code by `MLIRExecutionEngine`
4. **Symbol-resolved**: External `func.func private @mangled` declarations are resolved against the shared libraries passed to `create_jit()` — the user's compiled `.so` for C++ symbols, plus `libJLCS.so` for exception helpers (`jlcs_set_pending_exception`, `jlcs_has_pending_exception`, `jlcs_get_pending_exception`) and C++ runtime helpers (`__gxx_personality_v0`, `__cxa_begin_catch`, `__cxa_end_catch`).

The `lower_to_llvm()` function in `MLIRNative` drives the full lowering pass pipeline. MLIR dependencies used:

| MLIR Component | Role |
|----------------|------|
| `MLIRExecutionEngine` | JIT compilation and execution |
| `MLIRTargetLLVMIRExport` | MLIR module to LLVM IR translation |
| `MLIRLLVMToLLVMIRTranslation` | LLVM dialect lowering to native LLVM IR |

### AOT path: `ThunkBuilder.jl`

**Module:** `src/Builder/ThunkBuilder.jl`

When `aot_thunks = true` in `replibuild.toml`, the same JLCS IR that the JIT path would produce is instead lowered to LLVM IR text, written to disk, compiled to an object file via `llc`, and linked into a companion shared library named `<libname>_thunks.so`. The Julia wrapper then `ccall`s into this `.so` directly. There is no MLIR JIT at runtime — `libJLCS.so` is only needed for the initial codegen at build time, after which the AOT thunks are pure C-ABI shared library calls.

The AOT path reuses `JLCSIRGenerator.generate_jlcs_ir()` verbatim — there is no separate "AOT IR generator." The only differences between JIT and AOT are *when* the lowering happens (load time vs. build time) and *how* the resulting machine code is reached from Julia (cached symbol lookup vs. `ccall`).

## JIT manager

**Module:** `src/IRGen/JITManager.jl`

The JIT manager provides the runtime execution path for Tier 2 functions. It is a singleton (`GLOBAL_JIT`) that manages the MLIR context, JIT execution engine, and compiled symbol cache.

### Architecture

```
GLOBAL_JIT  (singleton)
───────────────────────
mlir_ctx          :: Ptr{Cvoid}                  # MLIR context
jit_engine        :: Ptr{Cvoid}                  # execution engine
compiled_symbols  :: Dict{String, Ptr{Cvoid}}    # lock-free hot path
vtable_info       :: VtableInfo                  # DWARF metadata
lock              :: ReentrantLock               # cold-path serialization
```

### Lock-free lookup (double-check pattern)

```
invoke("_mlir_ciface_foo_thunk", RetType, args...)
       │
       ▼
_lookup_cached(func_name)
       │
       ├── Fast path:  atomic Dict snapshot read  →  cache hit  →  return Ptr
       │
       └── Slow path:  lock  →  double-check  →  MLIRNative.lookup()
                              →  build new Dict (copy-on-write)
                              →  publish atomically  →  return Ptr
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

## Why MLIR for thunk generation

Thunk generation — emitting small wrapper functions that translate between two calling conventions — is not a novel technique. CFFI libraries, language runtimes, and FFI bridges have generated thunks for decades. What RepliBuild does that's worth describing in detail is *how* it uses MLIR for thunk generation, and which design properties fall out of that choice. The five points below are the ones that distinguish the approach in concrete terms.

### 1. DWARF byte offsets travel through the IR as typed attributes

The `!jlcs.c_struct<>` type carries field types and byte offsets as MLIR type parameters; `jlcs.marshal_arg` carries member types and Julia-side offsets as op attributes; `jlcs.vcall` carries vtable offset and slot index as integer attributes. None of these values are recomputed at any stage — they come from DWARF, ride through the dialect as structured attributes, and are consumed verbatim by the lowering passes to emit `getelementptr` instructions with exact byte offsets. The IR generator never holds a struct-layout calculator; it holds a translator from `DW_AT_data_member_location` integers to MLIR attribute integers.

### 2. ABI rules live in the C++ lowering pass, not in the Julia codegen

The Julia-side IR generator (`FunctionGen.jl`) emits roughly three lines of MLIR per packed-struct argument: a load of the Julia pointer, a `jlcs.marshal_arg` op, and a use of its result in the call. The actual field-by-field extractvalue/insertvalue/GEP/load/store sequence — and all of the platform-specific decisions about when to emit `sret`, when to use `byval`, when to pass scalars in registers — lives in `JLCSPasses.cpp` as MLIR `ConversionPattern` classes. This means the Julia codegen doesn't drift out of sync with platform ABIs when LLVM or the target triple changes; it stays declarative, and the lowering pass tracks the moving parts.

### 3. Exception safety is a typed operation, not a hand-written landing pad

`jlcs.try_call` is the same shape as `jlcs.ffe_call` — same `callee` attribute, same variadic args, same result type — but the lowering pass produces `llvm.invoke` + landing pad + exception buffer interaction instead of a plain `llvm.call`. The decision between the two ops is a single boolean check in the IR generator (`may_throw && !func_is_noexcept`), and the rest is structurally identical. Exception handling is not an afterthought retrofitted onto an existing FFI path; it's a peer operation that the dialect treats as first-class.

### 4. Ciface wrappers come for free via `llvm.emit_c_interface`

MLIR's standard LLVM lowering will emit a `_mlir_ciface_<name>` wrapper for any `func.func` with the `llvm.emit_c_interface` attribute. RepliBuild puts this attribute on every thunk, which gives it the `T ciface(void** args_ptr)` calling convention without hand-writing a trampoline. The Julia-side `JITManager.invoke` builds a `Ptr{Cvoid}[]` of `Ref`-converted argument pointers, calls the ciface, and reads the result. The trampoline that converts between this convention and the C++ ABI is generated by the MLIR framework itself — not by RepliBuild — which means the trampoline tracks any future ABI changes in MLIR's LLVM lowering without RepliBuild needing to track them.

### 5. JIT and AOT share the same IR generator

`JLCSIRGenerator.generate_jlcs_ir()` is the single source of MLIR text for both paths. The JIT path (`JITManager`) feeds the IR into MLIRExecutionEngine and looks up symbol pointers at runtime. The AOT path (`ThunkBuilder`) feeds the same IR into a write-to-disk + `llc` + `ld` pipeline that produces `<libname>_thunks.so`. The two paths differ only in *when* the lowering runs and *how* the resulting machine code is reached from Julia. There is no separate "AOT IR generator" that could drift from the JIT one — by construction, a function compiled AOT and a function compiled JIT execute identical machine code.

These five properties don't add up to a "novel thunk generator." They add up to a specific use of MLIR's framework features — typed dialects, ConversionPatterns, automatic ciface emission, ExecutionEngine — that lets RepliBuild keep ABI rules separate from codegen, keep JIT and AOT paths unified, and keep DWARF-derived data values flowing through the pipeline as structured IR rather than ad-hoc string substitution.

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
