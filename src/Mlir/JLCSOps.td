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
