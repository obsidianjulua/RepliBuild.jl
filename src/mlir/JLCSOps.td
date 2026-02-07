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
    DefaultValuedStrAttr<StrAttr, "\"\"">:$superType
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$typeName, "TypeAttr":$structType, "StringAttr":$superType)>
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
  let summary = "Call an external C function using FFE metadata";
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $results)";
}
