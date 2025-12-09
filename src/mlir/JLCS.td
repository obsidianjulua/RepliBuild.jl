// Top-level file for the JLCS dialect.

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def JLCS_Dialect : Dialect {
  let name = "jlcs";
  let cppNamespace = "::mlir::jlcs";
  let summary = "Julia C-Struct layout & FFE dialect";
  let description = [{
    Dialect that models C-layouted Julia structs and FFE (Foreign Function Execution)
    semantics for interop and ABI-aware lowering.
  }];
  let extraClassDeclaration = [{
    void registerTypes();
    void registerOps();
  }];
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class JLCS_Type<string name, string typeMnemonic> :
    TypeDef<JLCS_Dialect, name> {
  let mnemonic = typeMnemonic;
}

class JLCS_Op<string mnemonic, list<Trait> traits = []> :
    Op<JLCS_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

def CStructType : JLCS_Type<"CStruct", "c_struct"> {
  let summary = "A C-ABI-compatible struct with explicit field types and offsets.";
  let parameters = (ins
    "StringAttr":$juliaTypeName,
    ArrayRefParameter<"Type", "field types">:$fieldTypes,
    "ArrayAttr":$fieldOffsets
  );
  let assemblyFormat = "`<` $juliaTypeName `,` `[` $fieldTypes `]` `,` `[` $fieldOffsets `]` `>`";
  let genStorageClass = 1;
}

def ArrayViewType : JLCS_Type<"ArrayView", "array_view"> {
  let summary = "Universal strided array descriptor for cross-language arrays.";
  let parameters = (ins
    "Type":$elementType,
    "unsigned":$rank
  );
  let assemblyFormat = "`<` $elementType `,` $rank `>`";
  let genStorageClass = 1;
}

//===----------------------------------------------------------------------===//
// Operation Definitions
//===----------------------------------------------------------------------===//

def TypeInfoOp : JLCS_Op<"type_info", [Pure, IsolatedFromAbove]> {
  let summary = "Declares a CStruct type and its C++ base class mapping.";
  let arguments = (ins
    StrAttr   :$typeName,
    TypeAttr  :$structType,
    DefaultValuedStrAttr<StrAttr, "\"\"">:$superType
  );
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$typeName, "TypeAttr":$structType, "StringAttr":$superType)>
  ];
}

def GetFieldOp : JLCS_Op<"get_field"> {
  let summary = "Generic operation to read a field from a C-compatible struct.";
  let arguments = (ins
    AnyType:$structValue,
    I64Attr:$fieldOffset
  );
  let results = (outs AnyType:$result);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$structValue, "IntegerAttr":$fieldOffset, "Type":$resultType)>
  ];
}

def SetFieldOp : JLCS_Op<"set_field", [Pure]> {
  let summary = "Generic operation to write a field into a C-compatible struct.";
  let arguments = (ins
    AnyType:$structValue,
    AnyType:$newValue,
    I64Attr:$fieldOffset
  );
  let results = (outs);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$structValue, "Value":$newValue, "IntegerAttr":$fieldOffset)>
  ];
}

def VirtualCallOp : JLCS_Op<"vcall"> {
  let summary = "Call a C++ virtual method through vtable dispatch.";
  let arguments = (ins
    SymbolRefAttr:$class_name,
    Variadic<AnyType>:$args,
    I64Attr:$vtable_offset,
    I64Attr:$slot
  );
  let results = (outs Optional<AnyType>:$result);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$class_name, "ValueRange":$args,
                   "IntegerAttr":$vtable_offset, "IntegerAttr":$slot,
                   "Type":$resultType)>
  ];
}

def LoadArrayElementOp : JLCS_Op<"load_array_element"> {
  let summary = "Load element from strided array view.";
  let arguments = (ins
    AnyType:$view,
    Variadic<Index>:$indices
  );
  let results = (outs AnyType:$result);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$view, "ValueRange":$indices, "Type":$resultType)>
  ];
}

def StoreArrayElementOp : JLCS_Op<"store_array_element"> {
  let summary = "Store element to strided array view.";
  let arguments = (ins
    AnyType:$value,
    AnyType:$view,
    Variadic<Index>:$indices
  );
  let results = (outs);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$view, "ValueRange":$indices)>
  ];
}

def JLCS_FFECallOp : JLCS_Op<"ffe_call", [MemoryEffects<[]>]> {
  let summary = "Call an external C function using FFE metadata";
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$args attr-dict `:` type($args) `->` type($results)";
}

//===----------------------------------------------------------------------===//
// Interface Definitions
//===----------------------------------------------------------------------===//

// TODO: Re-enable Julia subtype interface
// Currently commented out due to TableGen syntax issues
// Will be added in a future iteration

// def JL_SubtypeInterface : TypeInterface<"JuliaSubtype"> {
//   let description = "Interface for types that are part of the Julia type hierarchy.";
//   let methods = [
//     InterfaceMethod<
//       "Get the supertype name",
//       "StringAttr", "getSupertypeName",
//       (ins "MLIRContext *":$ctx)
//     >
//   ];
// }