# JLCS: An MLIR Dialect for C++ ABI Marshalling in Foreign Language Runtimes

## Summary

JLCS (originally "Julia C-Struct") is an MLIR dialect that models C/C++ ABI-level interop for foreign language runtimes. It provides operations for struct layout marshalling across alignment boundaries, virtual method dispatch through vtable introspection, RAII-scoped object lifetime management, exception-safe foreign function invocation, and strided multi-dimensional array access. The dialect lowers entirely to the LLVM dialect via `ConversionPattern` implementations --- no DRR or PDLL. It is currently used to bridge a JIT-compiled language runtime (Julia) to C++ libraries, but the op semantics are language-agnostic: any runtime that needs to call into C++ code with correct ABI handling could use these ops as an intermediate representation between high-level binding descriptions and LLVM IR.

---

## Type System

### `!jlcs.c_struct` --- C-ABI-Compatible Struct

A struct type with explicit byte offsets per field and an explicit packed flag. This is the core type for representing layout mismatches between the host runtime's natural alignment and the target C/C++ ABI.

**TableGen definition** (`Types.td`):

```tablegen
def CStructType : JLCS_Type<"CStruct", "c_struct"> {
  let summary = "A C-ABI-compatible struct with explicit field types and offsets.";

  let parameters = (ins
    "StringAttr":$juliaTypeName,
    ArrayRefParameter<"Type", "field types">:$fieldTypes,
    "ArrayAttr":$fieldOffsets,
    "bool":$isPacked
  );

  let assemblyFormat = "`<` $juliaTypeName `,` `[` $fieldTypes `]` `,`"
                        " `[` $fieldOffsets `]` `,` `packed` `=` $isPacked `>`";

  let genStorageClass = 0;
  let storageClass = "CStructTypeStorage";
  let storageNamespace = "detail";
}
```

**MLIR syntax examples:**

```mlir
// Packed struct (no alignment padding) --- int a at offset 0, double b at offset 4
!jlcs.c_struct<"PackedPoint", [i32, f64], [[0 : i64, 4 : i64]], packed = true>

// Naturally aligned struct --- int a at offset 0, double b at offset 8
!jlcs.c_struct<"AlignedPoint", [i32, f64], [[0 : i64, 8 : i64]], packed = false>

// Class with vtable pointer + members
!jlcs.c_struct<"Shape", [!llvm.ptr, i32, f64],
               [[0 : i64, 8 : i64, 12 : i64]], packed = false>
```

**Type conversion** (in `LowerJLCSToLLVMPass`):

```cpp
typeConverter.addConversion([&](CStructType type) -> Type {
    SmallVector<Type> llvmFields;
    for (Type fieldType : type.getFieldTypes()) {
        llvmFields.push_back(typeConverter.convertType(fieldType));
    }
    return LLVM::LLVMStructType::getLiteral(
        &getContext(), llvmFields, type.getIsPacked());
});
```

`CStructType` lowers to `LLVM::LLVMStructType` with the packed flag preserved. The explicit byte offsets in `fieldOffsets` are consumed by `get_field` / `set_field` ops and do not appear in the lowered LLVM struct --- they guide GEP emission during lowering.

### `!jlcs.array_view` --- Universal Strided Array Descriptor

A parameterized view type for multi-dimensional strided arrays. The runtime layout is a fixed C struct:

```c
struct ArrayView {
    T*       data_ptr;     // offset 0
    int64_t* dims_ptr;     // offset 8
    int64_t* strides_ptr;  // offset 16
    int64_t  rank;         // offset 24
};
```

**TableGen definition** (`Types.td`):

```tablegen
def ArrayViewType : JLCS_Type<"ArrayView", "array_view"> {
  let summary = "Universal strided array descriptor for cross-language arrays.";

  let parameters = (ins
    "Type":$elementType,
    "unsigned":$rank
  );

  let assemblyFormat = "`<` $elementType `,` $rank `>`";

  let genStorageClass = 0;
  let storageClass = "ArrayViewTypeStorage";
  let storageNamespace = "detail";
}
```

**MLIR syntax examples:**

```mlir
!jlcs.array_view<f64, 2>       // 2D array of doubles
!jlcs.array_view<i32, 1>       // 1D array of ints
!jlcs.array_view<!jlcs.c_struct<"Particle", [f64, f64, f64], 
                  [[0:i64, 8:i64, 16:i64]], packed = false>, 3>  // 3D array of structs
```

---

## Operations

### Metadata & Field Access

#### `jlcs.type_info` --- Type Metadata Declaration

Declares a C++ class/struct type with optional inheritance and destructor metadata. Placed at module scope. Erased during lowering (pure metadata).

```tablegen
def TypeInfoOp : JLCS_Op<"type_info", [Pure, IsolatedFromAbove]> {
  let arguments = (ins
    StrAttr:$typeName,
    TypeAttr:$structType,
    DefaultValuedStrAttr<StrAttr, "\"\"">:$superType,
    DefaultValuedStrAttr<StrAttr, "\"\"">:$destructorName
  );
  let assemblyFormat = 
    "$typeName `,` $structType `,` $superType `,` $destructorName attr-dict";
}
```

```mlir
jlcs.type_info "Shape",
  !jlcs.c_struct<"Shape", [!llvm.ptr, i32, f64],
                 [[0 : i64, 8 : i64, 12 : i64]], packed = false>,
  "", "_ZN5ShapeD1Ev"

jlcs.type_info "Circle",
  !jlcs.c_struct<"Circle", [!llvm.ptr, i32, f64, f64],
                 [[0 : i64, 8 : i64, 12 : i64, 20 : i64]], packed = false>,
  "Shape", "_ZN6CircleD1Ev"
```

#### `jlcs.get_field` --- Byte-Offset Field Read

```tablegen
def GetFieldOp : JLCS_Op<"get_field", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins
    AnyType:$structValue,
    I64Attr:$fieldOffset
  );
  let results = (outs AnyType:$result);
}
```

Lowers to: `GEP(i8, structPtr, offset)` + `load`.

```mlir
%vptr = jlcs.get_field %obj_ptr { fieldOffset = 0 : i64 } : (!llvm.ptr) -> !llvm.ptr
%x    = jlcs.get_field %obj_ptr { fieldOffset = 8 : i64 } : (!llvm.ptr) -> i32
```

#### `jlcs.set_field` --- Byte-Offset Field Write

```tablegen
def SetFieldOp : JLCS_Op<"set_field", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins
    AnyType:$structValue,
    AnyType:$newValue,
    I64Attr:$fieldOffset
  );
  let results = (outs);
}
```

Lowers to: `GEP(i8, structPtr, offset)` + `store`.

### Virtual Method Dispatch

#### `jlcs.vcall` --- Vtable Indirect Call

```tablegen
def VirtualCallOp : JLCS_Op<"vcall", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins
    SymbolRefAttr:$class_name,
    Variadic<AnyType>:$args,      // first arg is always the object pointer
    I64Attr:$vtable_offset,       // byte offset of vptr in object layout
    I64Attr:$slot                 // vtable slot index
  );
  let results = (outs Optional<AnyType>:$result);
}
```

Lowers to a three-step sequence:

1. `load vptr` from object at `vtable_offset`
2. `GEP + load` function pointer from `vtable[slot]`
3. Indirect `llvm.call` through function pointer

```mlir
%area = jlcs.vcall @Shape::area(%obj)
  { vtable_offset = 0 : i64, slot = 2 : i64 } : (!llvm.ptr) -> f64
```

### Function Calls

#### `jlcs.ffe_call` --- Foreign Function Execution (Direct)

```tablegen
def FFECallOp : JLCS_Op<"ffe_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $results)";
}
```

Requires a `callee` attribute (FlatSymbolRefAttr) set on the op. During lowering, ABI coercion is applied:

- **Packed struct return**: allocated via `sret` (hidden first pointer arg, call returns void, result loaded from pointer)
- **Packed struct arguments**: passed via `byval` (alloca + store to stack slot, pass pointer)
- **Large non-packed struct return** (>128 bits on x86_64 SysV): uses `sret`
- The external function declaration's signature is **mutated in place** to match the coerced calling convention

```mlir
%r = jlcs.ffe_call %x, %y { callee = @_Z3addii }
  : (i32, i32) -> i32

jlcs.ffe_call %obj, %val { callee = @_ZN5Shape6setAreaEd }
  : (!llvm.ptr, f64) -> ()
```

#### `jlcs.try_call` --- Exception-Safe Foreign Call

```tablegen
def TryCallOp : JLCS_Op<"try_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $results)";
}
```

Same ABI coercion as `ffe_call`, but emits `llvm.invoke` instead of `llvm.call`. Block structure after lowering:

```
currentBlock:
  alloca %result_slot
  store zero -> %result_slot           // sentinel for exception path
  invoke @callee(%args) to ^ok unwind ^catch

^ok:
  store invoke_result -> %result_slot
  br ^merge

^catch:
  %lp = landingpad { ptr, i32 }       // catch-all (null filter)
  %exn = extractvalue %lp[0]
  call @__cxa_begin_catch(%exn)
  %msg = call @jlcs_catch_current_exception()
  call @__cxa_end_catch()
  br ^merge                            // result_slot still has zero sentinel

^merge:
  %result = load %result_slot
  // ... remainder of function ...
```

The host runtime checks `jlcs_has_pending_exception()` after every call and converts the thread-local message to a native exception.

Required runtime symbols (declared automatically):
- `__gxx_personality_v0` (personality function, set on enclosing `llvm.func`)
- `__cxa_begin_catch`, `__cxa_end_catch` (Itanium C++ ABI)
- `jlcs_set_pending_exception(const char*)`, `jlcs_catch_current_exception() -> const char*`

```mlir
%r = jlcs.try_call %arg0, %arg1 { callee = @_Z8might_throwii }
  : (i32, i32) -> i32
```

### Array Access

#### `jlcs.load_array_element` / `jlcs.store_array_element`

```tablegen
def LoadArrayElementOp : JLCS_Op<"load_array_element", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins
    AnyType:$view,
    Variadic<Index>:$indices
  );
  let results = (outs AnyType:$result);
}

def StoreArrayElementOp : JLCS_Op<"store_array_element", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins
    AnyType:$value,
    AnyType:$view,
    Variadic<Index>:$indices
  );
  let results = (outs);
}
```

Lowering computes linearized offset as `sum(index_i * stride_i)`, loads strides from the ArrayView descriptor at byte offset 16, and issues a single `GEP + load/store` on the data pointer at byte offset 0.

```mlir
%elem = jlcs.load_array_element %view[%i, %j]
  : !jlcs.array_view<f64, 2> -> f64

jlcs.store_array_element %val, %view[%i, %j]
  : f64, !jlcs.array_view<f64, 2>
```

### RAII Lifetime

#### `jlcs.ctor_call` / `jlcs.dtor_call`

```tablegen
def ConstructorCallOp : JLCS_Op<"ctor_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$args       // first arg = this pointer
  );
  let results = (outs);
  let assemblyFormat = 
    "$callee `(` $args `)` attr-dict `:` `(` type($args) `)` `->` `(` `)`";
}

def DestructorCallOp : JLCS_Op<"dtor_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    AnyType:$obj_ptr
  );
  let results = (outs);
  let assemblyFormat = 
    "$callee `(` $obj_ptr `)` attr-dict `:` `(` type($obj_ptr) `)` `->` `(` `)`";
}
```

Both lower to direct `llvm.call` to the mangled symbol. `dtor_call` enforces single-argument semantics (only `this`).

```mlir
jlcs.ctor_call @_ZN6CircleC1Ed(%ptr, %radius) : (!llvm.ptr, f64) -> ()
jlcs.dtor_call @_ZN6CircleD1Ev(%ptr) : (!llvm.ptr) -> ()
```

#### `jlcs.scope` / `jlcs.yield` --- Scoped RAII

```tablegen
def ScopeOp : JLCS_Op<"scope", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let arguments = (ins
    Variadic<AnyType>:$managed_ptrs,
    ArrayAttr:$destructors
  );
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);
  let assemblyFormat = 
    "`(` $managed_ptrs `:` type($managed_ptrs) `)` `dtors` `(`"
    " $destructors `)` $body attr-dict";
}

def YieldOp : JLCS_Op<"yield", [Terminator]> {
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}
```

Lowering: erases `yield`, inlines the body block before the scope op's position, then emits destructor calls **in reverse order** (matching C++ destruction semantics). The body region is **not** `IsolatedFromAbove` --- it can reference any SSA value from the enclosing scope.

```mlir
%alloca = llvm.alloca %one x !llvm.struct<(i32, f64)> : (i64) -> !llvm.ptr
jlcs.scope(%alloca : !llvm.ptr) dtors([@_ZN6CircleD1Ev]) {
  jlcs.ctor_call @_ZN6CircleC1Ed(%alloca, %radius) : (!llvm.ptr, f64) -> ()
  %area = jlcs.ffe_call %alloca { callee = @_ZN6Circle4areaEv }
    : (!llvm.ptr) -> f64
  jlcs.yield
}
// _ZN6CircleD1Ev(%alloca) emitted here by lowering
```

### ABI Marshalling

#### `jlcs.marshal_arg` --- Host-Aligned to C-Packed

```tablegen
def MarshalArgOp : JLCS_Op<"marshal_arg", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins
    AnyType:$srcPtr,
    ArrayAttr:$memberTypes,
    ArrayAttr:$juliaOffsets
  );
  let results = (outs AnyType:$result);
}
```

Reads each field from the host-runtime pointer at host-aligned byte offsets (`juliaOffsets`), then packs them into an LLVM packed struct via `undef` + `insertvalue` chain. Uses `alignment=1` loads to handle arbitrary host padding.

```mlir
%packed = jlcs.marshal_arg %ptr
  { memberTypes = [i32, f64], juliaOffsets = [0 : i64, 8 : i64] }
  : (!llvm.ptr) -> !llvm.struct<packed (i32, f64)>
```

#### `jlcs.marshal_ret` --- C-Packed to Host-Aligned

```tablegen
def MarshalRetOp : JLCS_Op<"marshal_ret", [Pure]> {
  let arguments = (ins
    AnyType:$packedValue,
    I64Attr:$numMembers
  );
  let results = (outs AnyType:$result);
}
```

Extracts each field from a packed struct value and inserts into a naturally-aligned struct via `extractvalue` + `insertvalue`.

```mlir
%aligned = jlcs.marshal_ret %packed { numMembers = 2 : i64 }
  : (!llvm.struct<packed (i32, f64)>) -> !llvm.struct<(i32, f64)>
```

---

## IR Examples

### Example 1: Simple Struct Field Access

**C++ source:**
```cpp
struct Point { int x; double y; };
int get_x(Point* p) { return p->x; }
```

**JLCS IR** (generated thunk):
```mlir
module {
  func.func private @_Z5get_xP5Point(!llvm.ptr) -> i32

  func.func @_Z5get_xP5Point_thunk(%args_ptr: !llvm.ptr) -> i32
      attributes { llvm.emit_c_interface } {
    %idx = arith.constant 0 : i64
    %arg_ptr = llvm.getelementptr %args_ptr[%idx]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr = llvm.load %arg_ptr : !llvm.ptr -> !llvm.ptr
    %p = llvm.load %val_ptr : !llvm.ptr -> !llvm.ptr

    %ret = jlcs.ffe_call %p { callee = @_Z5get_xP5Point }
      : (!llvm.ptr) -> i32
    return %ret : i32
  }
}
```

**After `jlcs-lower-to-llvm`:**
```mlir
llvm.func @_Z5get_xP5Point(!llvm.ptr) -> i32

llvm.func @_Z5get_xP5Point_thunk(%args_ptr: !llvm.ptr) -> i32
    attributes { llvm.emit_c_interface } {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.getelementptr %args_ptr[%0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
  %3 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
  %4 = llvm.call @_Z5get_xP5Point(%3) : (!llvm.ptr) -> i32
  llvm.return %4 : i32
}
```

**Emitted LLVM IR:**
```llvm
define i32 @_Z5get_xP5Point_thunk(ptr %args_ptr) {
  %1 = getelementptr ptr, ptr %args_ptr, i64 0
  %2 = load ptr, ptr %1
  %3 = load ptr, ptr %2
  %4 = call i32 @_Z5get_xP5Point(ptr %3)
  ret i32 %4
}
```

### Example 2: RAII Scope with Virtual Dispatch

**C++ source:**
```cpp
class Shape {
public:
    Shape(int id) : id_(id) {}
    virtual ~Shape();
    virtual double area() = 0;
protected:
    int id_;
};

class Circle : public Shape {
public:
    Circle(int id, double r) : Shape(id), radius_(r) {}
    ~Circle() override;
    double area() override { return 3.14159 * radius_ * radius_; }
private:
    double radius_;
};

double compute_area(int id, double radius) {
    Circle c(id, radius);
    return c.area();
}
```

**JLCS IR** (hand-written equivalent of what the pipeline produces; argument unpacking omitted --- same pattern as Example 1):
```mlir
module {
  // Type metadata
  jlcs.type_info "Shape",
    !jlcs.c_struct<"Shape", [!llvm.ptr, i32],
                   [[0 : i64, 8 : i64]], packed = false>,
    "", "_ZN5ShapeD1Ev"

  jlcs.type_info "Circle",
    !jlcs.c_struct<"Circle", [!llvm.ptr, i32, f64],
                   [[0 : i64, 8 : i64, 16 : i64]], packed = false>,
    "Shape", "_ZN6CircleD1Ev"

  // External symbols
  llvm.func @_ZN6CircleC1Eid(!llvm.ptr, i32, f64)
  llvm.func @_ZN6CircleD1Ev(!llvm.ptr)

  func.func @compute_area_thunk(%args_ptr: !llvm.ptr) -> f64
      attributes { llvm.emit_c_interface } {
    // ... argument unpacking (same pointer-chasing pattern as Example 1) ...
    // Produces: %id : i32, %radius : f64

    // Allocate Circle on stack (24 bytes: vptr + int + double)
    %one = arith.constant 1 : i64
    %obj = llvm.alloca %one x !llvm.struct<(!llvm.ptr, i32, f64)>
      : (i64) -> !llvm.ptr

    // RAII scope: construct, use, auto-destruct
    jlcs.scope(%obj : !llvm.ptr) dtors([@_ZN6CircleD1Ev]) {
      jlcs.ctor_call @_ZN6CircleC1Eid(%obj, %id, %radius)
        : (!llvm.ptr, i32, f64) -> ()

      // Virtual dispatch: area() is slot 2 in vtable
      %area = jlcs.vcall @Circle::area(%obj)
        { vtable_offset = 0 : i64, slot = 2 : i64 }
        : (!llvm.ptr) -> f64

      jlcs.yield
    }

    return %area : f64
  }
}
```

**After `jlcs-lower-to-llvm`** (type_info erased, scope inlined, vcall expanded):
```mlir
llvm.func @_ZN6CircleC1Eid(!llvm.ptr, i32, f64)
llvm.func @_ZN6CircleD1Ev(!llvm.ptr)

llvm.func @compute_area_thunk(%args_ptr: !llvm.ptr) -> f64
    attributes { llvm.emit_c_interface } {
  // ... argument unpacking (same as above) ...

  %obj = llvm.alloca %one x !llvm.struct<(!llvm.ptr, i32, f64)>
    : (i64) -> !llvm.ptr

  // Inlined scope body:
  // Constructor call
  llvm.call @_ZN6CircleC1Eid(%obj, %id, %radius) : (!llvm.ptr, i32, f64) -> ()

  // Virtual dispatch (vcall lowered):
  // 1. Load vptr from object at offset 0
  %vptr_addr = llvm.getelementptr %obj[%c0]
    : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %vptr = llvm.load %vptr_addr : !llvm.ptr -> !llvm.ptr

  // 2. Index vtable slot 2
  %slot = llvm.mlir.constant(2 : i64) : i64
  %fptr_addr = llvm.getelementptr %vptr[%slot]
    : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
  %fptr = llvm.load %fptr_addr : !llvm.ptr -> !llvm.ptr

  // 3. Indirect call
  %area = llvm.call %fptr(%obj) : (!llvm.ptr) -> f64

  // Scope exit: destructor emitted in reverse order
  llvm.call @_ZN6CircleD1Ev(%obj) : (!llvm.ptr) -> ()

  llvm.return %area : f64
}
```

**Emitted LLVM IR:**
```llvm
define double @compute_area_thunk(ptr %args_ptr) {
  ; ... argument unpacking ...
  %obj = alloca { ptr, i32, double }, i64 1
  call void @_ZN6CircleC1Eid(ptr %obj, i32 %id, double %radius)

  ; Virtual dispatch
  %vptr = load ptr, ptr %obj                     ; load vtable pointer
  %slot_addr = getelementptr ptr, ptr %vptr, i64 2
  %fptr = load ptr, ptr %slot_addr               ; load area() function pointer
  %area = call double %fptr(ptr %obj)            ; indirect call

  ; RAII cleanup
  call void @_ZN6CircleD1Ev(ptr %obj)
  ret double %area
}
```

### Example 3: Exception-Safe `try_call` with Packed Struct Return

**C++ source:**
```cpp
struct __attribute__((packed)) Result {
    int error_code;    // offset 0
    double value;      // offset 4 (packed, no padding)
};

Result compute(int input);  // may throw
```

**JLCS IR:**
```mlir
module {
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func @__cxa_begin_catch(!llvm.ptr) -> !llvm.ptr
  llvm.func @__cxa_end_catch()
  llvm.func @jlcs_set_pending_exception(!llvm.ptr)
  llvm.func @jlcs_catch_current_exception() -> !llvm.ptr

  func.func private @_Z7computei(i32) -> !llvm.struct<packed (i32, f64)>

  func.func @_Z7computei_thunk(%args_ptr: !llvm.ptr)
      -> !llvm.struct<(i32, f64)>
      attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : i64
    %arg_ptr = llvm.getelementptr %args_ptr[%c0]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %val_ptr = llvm.load %arg_ptr : !llvm.ptr -> !llvm.ptr
    %input = llvm.load %val_ptr : !llvm.ptr -> i32

    // Exception-safe call --- returns packed struct
    %ret_packed = jlcs.try_call %input { callee = @_Z7computei }
      : (i32) -> !llvm.struct<packed (i32, f64)>

    // Marshal packed return to aligned layout for host runtime
    %ret_aligned = jlcs.marshal_ret %ret_packed { numMembers = 2 : i64 }
      : (!llvm.struct<packed (i32, f64)>) -> !llvm.struct<(i32, f64)>

    return %ret_aligned : !llvm.struct<(i32, f64)>
  }
}
```

**After `jlcs-lower-to-llvm`** (try_call expanded to invoke + landing pad, marshal ops expanded):
```mlir
llvm.func @_Z7computei(!llvm.ptr) -> ()   // sret-coerced: packed struct via pointer

llvm.func @_Z7computei_thunk(%args_ptr: !llvm.ptr)
    -> !llvm.struct<(i32, f64)>
    attributes { llvm.emit_c_interface, personality = @__gxx_personality_v0 } {
  // ... argument unpacking ...

  // Allocate sret slot for packed return + result storage
  %one = llvm.mlir.constant(1 : i64) : i64
  %sret = llvm.alloca %one x !llvm.struct<packed (i32, f64)> : (i64) -> !llvm.ptr
  %result_slot = llvm.alloca %one x !llvm.struct<packed (i32, f64)> : (i64) -> !llvm.ptr
  %zero = llvm.mlir.zero : !llvm.struct<packed (i32, f64)>
  llvm.store %zero, %result_slot : !llvm.struct<packed (i32, f64)>, !llvm.ptr

  // invoke with sret convention
  llvm.invoke @_Z7computei(%sret, %input) to ^ok unwind ^catch
    : (!llvm.ptr, i32) -> ()

^ok:
  %packed_result = llvm.load %sret : !llvm.ptr -> !llvm.struct<packed (i32, f64)>
  llvm.store %packed_result, %result_slot
    : !llvm.struct<packed (i32, f64)>, !llvm.ptr
  llvm.br ^merge

^catch:
  %lp = llvm.landingpad (0, i32) : !llvm.struct<(!llvm.ptr, i32)>
  %exn = llvm.extractvalue %lp[0] : !llvm.struct<(!llvm.ptr, i32)>
  %_ = llvm.call @__cxa_begin_catch(%exn) : (!llvm.ptr) -> !llvm.ptr
  %msg = llvm.call @jlcs_catch_current_exception() : () -> !llvm.ptr
  llvm.call @__cxa_end_catch() : () -> ()
  llvm.br ^merge

^merge:
  %packed = llvm.load %result_slot
    : !llvm.ptr -> !llvm.struct<packed (i32, f64)>

  // marshal_ret lowered: extract from packed, insert into aligned
  %f0 = llvm.extractvalue %packed[0]
    : !llvm.struct<packed (i32, f64)> -> i32
  %f1 = llvm.extractvalue %packed[1]
    : !llvm.struct<packed (i32, f64)> -> f64
  %r0 = llvm.mlir.undef : !llvm.struct<(i32, f64)>
  %r1 = llvm.insertvalue %f0, %r0[0]
    : !llvm.struct<(i32, f64)>
  %r2 = llvm.insertvalue %f1, %r1[1]
    : !llvm.struct<(i32, f64)>

  llvm.return %r2 : !llvm.struct<(i32, f64)>
}
```

**Emitted LLVM IR:**
```llvm
define { i32, double } @_Z7computei_thunk(ptr %args_ptr)
    personality ptr @__gxx_personality_v0 {
entry:
  ; ... argument unpacking ...
  %sret = alloca <{ i32, double }>, i64 1
  %result_slot = alloca <{ i32, double }>, i64 1
  store <{ i32, double }> zeroinitializer, ptr %result_slot

  invoke void @_Z7computei(ptr %sret, i32 %input)
    to label %ok unwind label %catch

ok:
  %packed_ok = load <{ i32, double }>, ptr %sret
  store <{ i32, double }> %packed_ok, ptr %result_slot
  br label %merge

catch:
  %lp = landingpad { ptr, i32 } catch ptr null
  %exn = extractvalue { ptr, i32 } %lp, 0
  call ptr @__cxa_begin_catch(ptr %exn)
  call ptr @jlcs_catch_current_exception()
  call void @__cxa_end_catch()
  br label %merge

merge:
  %packed = load <{ i32, double }>, ptr %result_slot
  %ec = extractvalue <{ i32, double }> %packed, 0
  %val = extractvalue <{ i32, double }> %packed, 1
  %r0 = insertvalue { i32, double } undef, i32 %ec, 0
  %r1 = insertvalue { i32, double } %r0, double %val, 1
  ret { i32, double } %r1
}
```

---

## Open Design Questions

I am posting this for architectural feedback from people who work on MLIR dialects and/or C++ ABI lowering. The dialect is functional and tested against real C++ libraries (Lua, SQLite, pugixml, cJSON, etc.), but several design decisions feel under-specified or potentially wrong. Concrete feedback on any of the following would be valuable.

### 1. Should `ffe_call` and `try_call` be unified?

Currently these are two separate ops with identical signatures. The only difference is the lowering: `ffe_call` emits `llvm.call`, `try_call` emits `llvm.invoke` + landing pad. The IR generator selects between them based on a per-function `is_noexcept` attribute and a global `may_throw` flag derived from the source language.

An alternative design would be a single `jlcs.call` op with an optional `exception_handling = "none" | "itanium_cxx"` attribute. This would:

- Reduce op count and consolidate ABI coercion logic in the lowering pass (the two patterns currently share substantial implementation between `FFECallOpLowering` and `TryCallOpLowering`)
- Make it easier to add future exception models (SEH, setjmp/longjmp)
- Risk making the common case (no exceptions) carry dead attributes

**Question:** Is the two-op design defensible given the duplicated lowering, or should these be unified? If unified, should the exception attribute be an enum or a more general "calling convention" bundle?

### 2. Does `jlcs.scope` handle nested scopes and cross-scope references correctly?

`ScopeOp` uses `SingleBlockImplicitTerminator<"YieldOp">` and is not `IsolatedFromAbove`. During lowering, the body is inlined into the parent block and destructors are appended in reverse order. This works for flat scopes, but:

- **Nested scopes**: If a `jlcs.scope` contains another `jlcs.scope`, the inner scope is lowered first (assuming bottom-up pattern application). The inner destructors are emitted, then the outer body continues, then the outer destructors fire. This appears correct for normal control flow but has not been formally verified.

- **Cross-scope SSA references**: Because the body is not isolated, values defined inside the scope body can potentially be referenced after the scope (if they dominate). After inlining, this is fine --- but the *semantic contract* is that the scope represents a lifetime boundary. There is no verifier check that prevents post-scope use of scope-local values.

- **Scope results**: Currently `ScopeOp` has no results. If the body needs to produce a value (e.g., the result of a method call on a scoped object, as in Example 2), that value must be defined inside the body but used after the scope. This works because inlining merges the blocks, but it feels like an accidental property of the lowering rather than a deliberate design.

**Question:** Should `ScopeOp` support result values (via `YieldOp` carrying values out)? Should there be a verifier that prevents escape of scope-managed pointers?

### 3. Is destructor emission order correct under exception unwind?

The current `ScopeOpLowering` unconditionally emits destructors after the inlined body, on the normal control flow path only. If the body contains a `try_call` that throws:

1. `try_call` catches the exception in its landing pad
2. The landing pad stores the exception message and branches to the merge block
3. Control continues past the scope
4. The scope's destructors fire

This means destructors **do** run after a caught exception, which is correct. However:

- **Uncaught exceptions**: If a future lowering path needs cleanup semantics (destructors run even when the function itself unwinds), the current design has no mechanism for it. There is no `funclet` or `cleanuppad` emission.
- **Multiple scoped objects with interleaved `try_call`s**: If scope A contains a `try_call` that throws, scope A's destructors run. But if scope B is *nested inside* scope A's body, and the `try_call` is between B's end and A's end, then B's destructors have already run (they were inlined earlier), but A's destructors fire after the catch. This appears correct but the ordering depends entirely on pattern application order.
- **C++ EH contract**: In real C++ unwinding, destructors run during stack unwinding via cleanup landing pads. JLCS does not model this --- it catches-and-continues rather than unwinds. This means JLCS cannot correctly handle cases where the destructor itself might throw, or where RAII objects need to be cleaned up during unwinding through frames that don't catch.

**Question:** Is catch-and-continue sufficient for the FFI use case, or should scope lowering emit cleanup landing pads for correctness? If cleanup pads are needed, should `ScopeOp` grow a cleanup region?

### 4. C++ ABI patterns not currently representable

The dialect was designed for the common case first --- single-inheritance classes, Itanium ABI, x86_64 --- and the following gaps are deliberate scoping decisions, not oversights. The question is which are worth closing at the dialect level vs. handling in the IR generation layer above.

The following C++ constructs have no corresponding JLCS ops or are handled outside the dialect:

- **Multiple inheritance**: `vcall` uses a single `vtable_offset` to locate the vptr. With multiple inheritance, an object can have multiple vptrs at different offsets, and `this`-pointer adjustment is needed when calling through a secondary base's vtable. The current `vtable_offset` attribute could point at a secondary vptr, but there is no mechanism for the `this` adjustment (adding a fixed offset to the object pointer before the call).

- **Virtual base classes**: Virtual inheritance adds an indirection through the VTT (virtual table table) or vbase offset stored in the vtable. None of this is modeled.

- **Move semantics**: `ctor_call` can call a move constructor, but there is no semantic distinction between copy and move at the op level. If the dialect needed to reason about ownership transfer (e.g., for optimization or verification), there is no way to express it.

- **Non-trivial return types**: When a C++ function returns a class with a non-trivial destructor, the caller is responsible for providing storage (sret) and eventually destroying the returned object. Currently, `ffe_call`/`try_call` handle the sret ABI coercion, but there is no automatic destructor registration for the returned value. The host runtime must manually manage this.

- **Thiscall and other calling conventions**: The dialect assumes Itanium ABI (x86_64 SysV). MSVC ABI (`thiscall`, different vptr layout, different exception handling via SEH) is not supported.

- **Covariant return types**: Virtual methods with covariant returns require thunks in the vtable that adjust the return pointer. `vcall` loads and calls the vtable slot directly --- if the slot contains a covariant thunk, it works accidentally (the thunk handles it), but the dialect has no model for this.

**Question:** Which of these gaps are worth closing in the dialect itself vs. handling in the IR generation layer? Multiple inheritance `this`-adjustment seems like it should be an op-level concern; move semantics probably should not.

---

*This dialect is part of a larger toolchain that compiles C/C++ libraries, extracts DWARF metadata, and generates type-safe bindings for a host runtime. The MLIR layer sits between DWARF-derived type information and LLVM IR execution. Feedback on the op design, type system, and lowering strategy is welcome.*
