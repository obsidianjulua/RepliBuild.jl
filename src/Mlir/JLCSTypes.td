// include/JLCSDialect/JLCSTypes.td

include "JLCSDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for all types in the JLCS dialect
class JLCS_Type<string name, string typeMnemonic> :
TypeDef<JLCS_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// ====================================================================
// The Generalized Type: CStruct
// ====================================================================

def CStructType : JLCS_Type<"CStruct", "c_struct"> {
  let summary = "A C-ABI-compatible struct with explicit field types and offsets.";

  // Parameters define the *stored state* of the type.
  let parameters = (ins
  // 1. The fully qualified Julia type name (e.g., "MyModule.Outer")
  "StringAttr":$juliaTypeName,
  // 2. The *flattened* field types (including nested struct fields)
  ArrayRefParameter<"Type", "field types">:$fieldTypes,
  // 3. The explicit byte offsets for each field (Crucial for C-ABI compatibility)
  "ArrayAttr":$fieldOffsets
  );

  // Assembly format for MLIR text: !jlcs.c_struct<"Name", [T1, T2], [O1, O2]>
  let assemblyFormat = "`<` $juliaTypeName `,` `[` $fieldTypes `]` `,` `[` $fieldOffsets `]` `>`";
}
