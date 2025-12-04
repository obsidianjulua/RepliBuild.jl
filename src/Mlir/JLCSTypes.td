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

  // Generate storage class
  let genStorageClass = 1;
}

// ====================================================================
// Universal Array View Type
// ====================================================================

def ArrayViewType : JLCS_Type<"ArrayView", "array_view"> {
  let summary = "Universal strided array descriptor for cross-language arrays.";
  let description = [{
    Canonical representation of multi-dimensional arrays compatible with:
    - Julia Array{T,N}
    - NumPy ndarray
    - C++ strided arrays
    - Rust slices

    Runtime memory layout (as C struct):
      struct ArrayView {
        T* data_ptr;        // offset 0:  pointer to element data
        int64_t* dims_ptr;  // offset 8:  pointer to dimension sizes
        int64_t* strides_ptr; // offset 16: pointer to stride values (in elements)
        int64_t rank;       // offset 24: number of dimensions
      };

    This type represents a *view* into array data with explicit strides,
    enabling zero-copy interop across language boundaries.
  }];

  let parameters = (ins
    "Type":$elementType,  // Element type (f64, i32, !jlcs.c_struct<...>, etc.)
    "unsigned":$rank      // Number of dimensions (compile-time constant)
  );

  // Assembly format: !jlcs.array_view<f64, 3>
  let assemblyFormat = "`<` $elementType `,` $rank `>`";

  // Generate storage class
  let genStorageClass = 1;
}
