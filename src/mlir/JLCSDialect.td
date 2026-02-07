// include/JLCSDialect/JLCSDialect.td

#ifndef JLCS_DIALECT_TD
#define JLCS_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

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

  // LLVM 21: Use auto-generated type parser/printer
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerOps();
  }];
}

#endif // JLCS_DIALECT_TD
