// include/JLCSDialect/JLCSDialect.td

#ifndef JULIA_C_STRUCT_DIALECT_TD
#define JULIA_C_STRUCT_DIALECT_TD

include "mlir/IR/DialectBase.td"

// Define the Dialect
def JLCS_Dialect : Dialect
{
    let name = "jlcs"; // The namespace used in MLIR code (e.g., jlcs.type_info)
    let summary = "Julia C-Struct Layout and FFI Dialect.";
    let description = [{
    The Julia C-Struct Dialect (JLCS) is designed to model Julia struct
    types in a C-ABI-compatible, flattened layout for interoperability
    with C++ and FFI tools like Julia ccall.
  }];
    let cppNamespace = "::mlir::jlcs";
}

#endif // JULIA_C_STRUCT_DIALECT_TD
