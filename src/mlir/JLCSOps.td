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

  let arguments = (ins
    // The Julia name of the type
    StrAttr   :$typeName,
    // The MLIR Type definition (must be CStructType)
    TypeAttr  :$structType,
    // The C++ class name of the *supertype* (for inheritance)
    DefaultValuedStrAttr<StrAttr, "\"\"">:$superType,
    // Mangled name of the C++ destructor (e.g., "_ZN4BaseD1Ev")
    DefaultValuedStrAttr<StrAttr, "\"\"">:$destructorName
  );

  // Custom assembly: jlcs.type_info "TypeName", !jlcs.c_struct<...>, "SuperType", "DtorName"
  let assemblyFormat = "$typeName `,` $structType `,` $superType `,` $destructorName attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$typeName, "TypeAttr":$structType, "StringAttr":$superType, "StringAttr":$destructorName)>
  ];

  let extraClassDeclaration = [{
    TypeInfoOp(::mlir::Operation *op) : Op(op) {}
  }];
}

// 2. Generic Field Access Op (Universal Getter)
def GetFieldOp : JLCS_Op<"get_field"> {
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
def SetFieldOp : JLCS_Op<"set_field", [Pure]> {
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
def VirtualCallOp : JLCS_Op<"vcall"> {
  let summary = "Call a C++ virtual method through vtable dispatch.";
  let description = [{
    Calls a C++ virtual method by:
    1. Reading the vtable pointer from the object (at vtable_offset)
    2. Loading the function pointer from vtable[slot]
    3. Calling the function with the object pointer + arguments

    Example:
    ```mlir
    %result = jlcs.vcall @Base::foo(%obj)
      { vtable_offset = 0 : i64, slot = 0 : i64 }
      : (!llvm.ptr) -> i32
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$class_name,    // Class name (e.g., @Base)
    Variadic<AnyType>:$args,      // Arguments (first is always object pointer)
    I64Attr:$vtable_offset,       // Offset of vptr in object
    I64Attr:$slot                 // Vtable slot index
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
}

// ====================================================================
// Array Operations
// ====================================================================

// 5. Load Array Element Op (Strided Array Access)
def LoadArrayElementOp : JLCS_Op<"load_array_element"> {
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
def StoreArrayElementOp : JLCS_Op<"store_array_element"> {
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
def FFECallOp : JLCS_Op<"ffe_call", [MemoryEffects<[]>]> {
  let summary = "Call an external function using FFE metadata";
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $results)";
}

// ====================================================================
// Exception-Handling Call Operation
// ====================================================================

// 12. Try Call Op (FFE Call with C++ exception catching)
def TryCallOp : JLCS_Op<"try_call", [MemoryEffects<[]>]> {
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
def ConstructorCallOp : JLCS_Op<"ctor_call"> {
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
def DestructorCallOp : JLCS_Op<"dtor_call"> {
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
}
