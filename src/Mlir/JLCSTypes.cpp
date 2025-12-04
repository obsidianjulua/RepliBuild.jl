//===- JLCSTypes.cpp - JLCS dialect types --------------------------------===//
//
// Type system implementation for JLCS dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "JLCSTypes.h"
#include "JLCSDialect.h"

using namespace mlir;
using namespace mlir::jlcs;

//===----------------------------------------------------------------------===//
// CStructType implementation
//===----------------------------------------------------------------------===//

// Note: Type verification is handled by TableGen-generated code in MLIR 21+
// Custom verification can be added via TypeDef::genVerifyDecl if needed

//===----------------------------------------------------------------------===//
// TableGen generated type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type Registration
//===----------------------------------------------------------------------===//

// Called from JLCSDialect constructor to register types
void JLCSDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "JLCSTypes.cpp.inc"
  >();
}
