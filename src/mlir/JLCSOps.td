// include/JLCSDialect/JLCSOps.td

include "JLCSDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

class JLCS_Op<string mnemonic, list<Trait> traits = []> :
    Op<JLCS_Dialect, mnemonic, traits>;

// ====================================================================
// Operations for Metadata and Field Access
// ====================================================================

// 1. Metadata Declaration Op (Global Information)
// This Op is placed in the module's top region and holds the C++ inheritance mapping.
def TypeInfoOp : JLCS_Op<"type_info", [Pure, IsolatedFromAbove]> {
  let summary = "Declares a CStruct type and its C++ base class mapping.";
  let description = [{
    Metadata op (erased at lowering). `superType` names the primary base
    for single-inheritance consumers. Under multiple inheritance the full
    picture lives in the paired `baseNames`/`baseOffsets` arrays: entry i
    says base class baseNames[i]'s subobject starts baseOffsets[i] bytes
    into the object (Itanium static layout). The arrays default to empty,
    so single-inheritance IR is unchanged; when present, baseNames[0] is
    the primary base (offset 0) and `superType` matches it. Virtual bases
    are not representable here — producers must reject them upstream.
  }];

  let arguments = (ins
    // The Julia name of the type
    StrAttr   :$typeName,
    // The MLIR Type definition (must be CStructType)
    TypeAttr  :$structType,
    // The C++ class name of the *supertype* (for inheritance)
    DefaultValuedStrAttr<StrAttr, "\"\"">:$superType,
    // Mangled name of the C++ destructor (e.g., "_ZN4BaseD1Ev")
    DefaultValuedStrAttr<StrAttr, "\"\"">:$destructorName,
    // Multiple-inheritance base table: names paired 1:1 with byte offsets
    // of each base subobject in the derived object. Ride in attr-dict.
    // NB: ArrayAttr's constBuilderCall already wraps the default in
    // $_builder.getArrayAttr(...) — the default here is the *contents*.
    DefaultValuedAttr<ArrayAttr, "{}">:$baseNames,
    DefaultValuedAttr<ArrayAttr, "{}">:$baseOffsets
  );

  // Custom assembly: jlcs.type_info "TypeName", !jlcs.c_struct<...>, "SuperType", "DtorName"
  // (baseNames/baseOffsets print in the trailing attr-dict when non-empty)
  let assemblyFormat = "$typeName `,` $structType `,` $superType `,` $destructorName attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$typeName, "TypeAttr":$structType, "StringAttr":$superType, "StringAttr":$destructorName)>
  ];

  let extraClassDeclaration = [{
    TypeInfoOp(::mlir::Operation *op) : Op(op) {}
  }];

  // baseNames/baseOffsets must pair 1:1 (same idiom as marshal_arg's
  // memberTypes/juliaOffsets — consumers index one by the other's position).
  let hasVerifier = 1;
}

// 2. Generic Field Access Op (Universal Getter)
def GetFieldOp : JLCS_Op<"get_field", [MemoryEffects<[MemRead]>]> {
  let summary = "Generic operation to read a field from a C-compatible struct.";

  let arguments = (ins
    AnyType:$structValue, // The struct itself (a memory reference or value)
    I64Attr:$fieldOffset  // The byte offset of the field, read from the CStructType
  );

  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$structValue, "IntegerAttr":$fieldOffset, "Type":$resultType)>
  ];

  let extraClassDeclaration = [{
    GetFieldOp(::mlir::Operation *op) : Op(op) {}
  }];
}

// 3. Generic Field Mutate Op (Universal Setter)
def SetFieldOp : JLCS_Op<"set_field", [MemoryEffects<[MemWrite]>]> {
  let summary = "Generic operation to write a field into a C-compatible struct.";

  let arguments = (ins
    AnyType:$structValue,  // The struct memory reference
    AnyType:$newValue,     // The new value to set
    I64Attr:$fieldOffset   // The byte offset of the field
  );

  // set_field does not return a value
  let results = (outs);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$structValue, "Value":$newValue, "IntegerAttr":$fieldOffset)>
  ];

  let extraClassDeclaration = [{
    SetFieldOp(::mlir::Operation *op) : Op(op) {}
  }];
}

// ====================================================================
// Virtual Method Call Operation
// ====================================================================

// 4. Virtual Call Op (Call C++ virtual method via vtable)
def VirtualCallOp : JLCS_Op<"vcall", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call a C++ virtual method through vtable dispatch.";
  let description = [{
    Calls a C++ virtual method by:
    1. Reading the vtable pointer from the object (at vtable_offset)
    2. Loading the function pointer from vtable[slot]
    3. Calling the function with the (possibly adjusted) object pointer
       + remaining arguments

    Both offsets are byte offsets relative to the ORIGINAL object pointer
    in args[0]. `vtable_offset` locates the vptr to read; `this_offset`
    is added to args[0] before the call so the callee receives a pointer
    to the correct base subobject. For a method inherited from a
    non-primary base under multiple inheritance (Itanium ABI), both equal
    the base subobject's offset in the derived object; for single
    inheritance / primary base both are 0. Virtual bases are NOT modeled
    (their offsets live in the vtable, not the static layout).

    Example (secondary base at byte 16):
    ```mlir
    %result = jlcs.vcall @Derived::via_Base2(%obj)
      { vtable_offset = 16 : i64, this_offset = 16 : i64, slot = 0 : i64 }
      : (!llvm.ptr) -> i32
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$class_name,    // Class name (e.g., @Base)
    Variadic<AnyType>:$args,      // Arguments (first is always object pointer)
    I64Attr:$vtable_offset,       // Offset of vptr in object (from args[0])
    I64Attr:$slot,                // Vtable slot index
    // Byte adjustment applied to args[0] before the call (`this` fixup for
    // methods dispatched through a non-primary base). Defaults to 0 so all
    // pre-MI IR keeps its exact semantics.
    DefaultValuedAttr<I64Attr, "0">:$this_offset,
    // When present, lower as invoke + landing pad (same sentinel-continue EH
    // model as jlcs.try_call: exception → jlcs_set_pending_exception, zero
    // result). Absent = plain indirect call, the pre-existing lowering.
    // NOTE: the EH path (like the plain path) does no sret/packed ABI
    // coercion — producers must only set this on scalar/pointer signatures.
    UnitAttr:$may_throw
  );

  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$class_name, "ValueRange":$args,
                   "IntegerAttr":$vtable_offset, "IntegerAttr":$slot,
                   "Type":$resultType)>
  ];

  let extraClassDeclaration = [{
    VirtualCallOp(::mlir::Operation *op) : Op(op) {}

    // Helper to get the object pointer (first argument)
    Value getObject() { return getArgs()[0]; }
  }];

  // The lowering unconditionally reads args[0] as the object pointer;
  // zero-arg IR previously only failed inside the conversion pattern.
  let hasVerifier = 1;
}

// ====================================================================
// Array Operations
// ====================================================================

// 5. Load Array Element Op (Strided Array Access)
def LoadArrayElementOp : JLCS_Op<"load_array_element", [MemoryEffects<[MemRead]>]> {
  let summary = "Load element from strided array view.";
  let description = [{
    Loads an element from a multi-dimensional array using strided indexing.
    Computes the linearized offset as: sum(index_i * stride_i) for all dimensions.
    Then performs: data_ptr[linear_offset]

    The array view must be a pointer to an ArrayView struct containing:
    - data_ptr (offset 0)
    - dims_ptr (offset 8)
    - strides_ptr (offset 16)
    - rank (offset 24)

    Example:
    ```mlir
    %elem = jlcs.load_array_element %view[%i, %j, %k]
      : !jlcs.array_view<f64, 3> -> f64
    ```
  }];

  let arguments = (ins
    AnyType:$view,               // Pointer to ArrayView struct
    Variadic<Index>:$indices     // Multi-dimensional indices (must match rank)
  );

  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$view, "ValueRange":$indices, "Type":$resultType)>
  ];

  let extraClassDeclaration = [{
    LoadArrayElementOp(::mlir::Operation *op) : Op(op) {}
  }];
}

// 6. Store Array Element Op (Strided Array Mutation)
def StoreArrayElementOp : JLCS_Op<"store_array_element", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store element to strided array view.";
  let description = [{
    Stores an element to a multi-dimensional array using strided indexing.
    Computes the linearized offset as: sum(index_i * stride_i) for all dimensions.
    Then performs: data_ptr[linear_offset] = value

    Example:
    ```mlir
    jlcs.store_array_element %value, %view[%i, %j, %k]
      : f64, !jlcs.array_view<f64, 3>
    ```
  }];

  let arguments = (ins
    AnyType:$value,              // Value to store
    AnyType:$view,               // Pointer to ArrayView struct
    Variadic<Index>:$indices     // Multi-dimensional indices
  );

  let results = (outs);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$view, "ValueRange":$indices)>
  ];

  let extraClassDeclaration = [{
    StoreArrayElementOp(::mlir::Operation *op) : Op(op) {}
  }];
}

// 7. FFE Call Op (Foreign Function Execution)
def FFECallOp : JLCS_Op<"ffe_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call an external function using FFE metadata";
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $results)";
}

// ====================================================================
// Exception-Handling Call Operation
// ====================================================================

// 12. Try Call Op (FFE Call with C++ exception catching)
def TryCallOp : JLCS_Op<"try_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call external function with C++ exception catching";
  let description = [{
    Like ffe_call but emits invoke + landing pad to catch C++ exceptions.
    On exception: stores error message in thread-local buffer via
    jlcs_set_pending_exception(), returns zero/null sentinel value.
    The caller (Julia side) checks jlcs_has_pending_exception() after
    the call and throws a CxxException if set.

    Example:
    ```mlir
    %result = jlcs.try_call %arg0, %arg1
      { callee = @_Z3fooi }
      : (i32, f64) -> i32
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $results)";
}

// ====================================================================
// RAII Operations (Constructor / Destructor)
// ====================================================================

// 8. Constructor Call Op
def ConstructorCallOp : JLCS_Op<"ctor_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call a C++ constructor on an object pointer.";
  let description = [{
    Calls a C++ constructor by invoking the named symbol with the object
    pointer as the first argument (the implicit `this`), followed by any
    additional constructor parameters.

    The first argument must always be a pointer to the memory where the
    object is being constructed (stack alloca or heap allocation).

    Example:
    ```mlir
    jlcs.ctor_call @_ZN4BaseC1Ei(%ptr, %val) : (!llvm.ptr, i32) -> ()
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,    // Mangled constructor symbol
    Variadic<AnyType>:$args       // First arg = this pointer, rest = ctor params
  );

  let results = (outs);

  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` `(` type($args) `)` `->` `(` `)`";
}

// 9. Destructor Call Op
def DestructorCallOp : JLCS_Op<"dtor_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call a C++ destructor on an object pointer.";
  let description = [{
    Calls a C++ destructor by invoking the named symbol with the object
    pointer as the sole argument (the implicit `this`).

    This operation guarantees single-argument semantics — the destructor
    takes only the `this` pointer and returns void.

    Example:
    ```mlir
    jlcs.dtor_call @_ZN4BaseD1Ev(%ptr) : (!llvm.ptr) -> ()
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,    // Mangled destructor symbol
    AnyType:$obj_ptr              // Object pointer (this)
  );

  let results = (outs);

  let assemblyFormat = "$callee `(` $obj_ptr `)` attr-dict `:` `(` type($obj_ptr) `)` `->` `(` `)`";
}

// ====================================================================
// RAII Scope Operations
// ====================================================================

// 10. Yield Op (Scope terminator)
def YieldOp : JLCS_Op<"yield", [Terminator]> {
  let summary = "Terminator for jlcs.scope regions.";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

// 11. Scope Op (RAII scoped lifetime)
def ScopeOp : JLCS_Op<"scope", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "RAII scope that guarantees destructor calls at scope exit.";
  let description = [{
    Defines a scoped lifetime for C++ objects. The scope takes pointers to
    managed objects as operands and an array of destructor symbols. When the
    scope exits (during lowering), destructor calls are emitted in reverse
    order — matching C++ destruction semantics.

    The body region can reference any SSA value visible in the enclosing
    scope (not IsolatedFromAbove).

    Example:
    ```mlir
    %alloca = llvm.alloca 1 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
    jlcs.scope(%alloca : !llvm.ptr) dtors([@_ZN4BaseD1Ev]) {
      jlcs.ctor_call @_ZN4BaseC1Ev(%alloca) : (!llvm.ptr) -> ()
      // ... use object ...
      jlcs.yield
    }
    // destructor called automatically here
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$managed_ptrs,   // Object pointers whose lifetime is managed
    ArrayAttr:$destructors             // Matching destructor symbols (FlatSymbolRefAttrs)
  );

  let regions = (region SizedRegion<1>:$body);

  let results = (outs);

  let assemblyFormat = "`(` $managed_ptrs `:` type($managed_ptrs) `)` `dtors` `(` $destructors `)` $body attr-dict";

  // Arity mismatch between managed_ptrs and destructors used to SIGSEGV
  // inside ScopeOpLowering::emitDestructors (indexes managedPtrs by the
  // destructor position). Producers co-generate the two lists; hand-written
  // IR now gets a diagnostic instead of a crash.
  let hasVerifier = 1;
}

// ====================================================================
// ABI Marshalling Operations
// ====================================================================

// 13. Marshal Arg Op (Julia-aligned pointer → C-packed struct value)
def MarshalArgOp : JLCS_Op<"marshal_arg", [MemoryEffects<[MemRead]>]> {
  let summary = "Pack aligned Julia struct into C-packed layout for FFI call.";
  let description = [{
    Reads an aligned Julia struct from a pointer, loading each field at its
    Julia-aligned byte offset, and assembles them into a packed LLVM struct
    suitable for passing to a C/C++ function.

    The operation takes a pointer to the Julia-side aligned struct and produces
    the packed struct value. The attributes carry the layout mismatch info:
    - memberTypes: the MLIR types of each struct member (for typed loads)
    - juliaOffsets: the byte offset of each member in the Julia-aligned layout

    Example:
    ```mlir
    %packed = jlcs.marshal_arg %ptr
      { memberTypes = [i32, f64], juliaOffsets = [0 : i64, 8 : i64] }
      : (!llvm.ptr) -> !llvm.struct<packed (i32, f64)>
    ```
  }];

  let arguments = (ins
    AnyType:$srcPtr,            // Pointer to Julia-aligned struct
    ArrayAttr:$memberTypes,     // Array of TypeAttr (field types for typed loads)
    ArrayAttr:$juliaOffsets      // Array of IntegerAttr i64 (byte offsets in Julia layout)
  );

  let results = (outs AnyType:$result);  // Packed struct value

  let assemblyFormat = "$srcPtr attr-dict `:` functional-type($srcPtr, $result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$srcPtr, "ArrayAttr":$memberTypes,
                   "ArrayAttr":$juliaOffsets, "Type":$resultType)>
  ];

  let extraClassDeclaration = [{
    MarshalArgOp(::mlir::Operation *op) : Op(op) {}
  }];

  // memberTypes/juliaOffsets arity mismatch used to SIGSEGV during lowering
  // (the field loop indexes juliaOffsets by memberTypes position). Diagnose
  // instead; also pin the element kinds the lowering assumes.
  let hasVerifier = 1;
}

// 14. Marshal Ret Op (C-packed struct value → Julia-aligned struct value)
def MarshalRetOp : JLCS_Op<"marshal_ret", [Pure]> {
  let summary = "Unpack C-packed struct return into Julia-aligned layout.";
  let description = [{
    Takes a packed struct value returned from a C/C++ function and repacks
    its members into a Julia-aligned struct layout. Each member is extracted
    from the packed struct and inserted into the aligned struct at the
    corresponding position.

    Example:
    ```mlir
    %aligned = jlcs.marshal_ret %packed
      { numMembers = 2 : i64 }
      : (!llvm.struct<packed (i32, f64)>) -> !llvm.struct<(i32, f64)>
    ```
  }];

  let arguments = (ins
    AnyType:$packedValue,       // Packed struct from C call
    I64Attr:$numMembers         // Number of struct members (loop bound for lowering)
  );

  let results = (outs AnyType:$result);  // Aligned struct value

  let assemblyFormat = "$packedValue attr-dict `:` functional-type($packedValue, $result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$packedValue, "IntegerAttr":$numMembers,
                   "Type":$resultType)>
  ];

  let extraClassDeclaration = [{
    MarshalRetOp(::mlir::Operation *op) : Op(op) {}
  }];
}
