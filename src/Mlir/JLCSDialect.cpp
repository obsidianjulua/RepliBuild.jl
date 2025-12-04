//===- JLCSDialect.cpp - JLCS dialect implementation ---------------------===//
//
// Julia C-Struct Dialect for FFI type representation
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"

// Include generated dialect declarations
#include "JLCSDialect.h.inc"

// Include generated type and operation definitions
#include "JLCSTypes.h"
#include "JLCSOps.h"

using namespace mlir;
using namespace mlir::jlcs;

//===----------------------------------------------------------------------===//
// JLCS Dialect
//===----------------------------------------------------------------------===//

#include "JLCSDialect.cpp.inc"

void JLCSDialect::initialize() {
  // Register operations
  addOperations<
#define GET_OP_LIST
#include "JLCSOps.cpp.inc"
  >();

  // Register types (defined in JLCSTypes.cpp to avoid circular deps)
  registerTypes();
}

//===----------------------------------------------------------------------===//
// Dialect Registration (C API for Julia)
//===----------------------------------------------------------------------===//

extern "C" {

// Register dialect with an MLIR context
void registerJLCSDialect(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  mlir::DialectRegistry registry;
  registry.insert<mlir::jlcs::JLCSDialect>();
  ctx->appendDialectRegistry(registry);
  ctx->loadDialect<mlir::jlcs::JLCSDialect>();
}

} // extern "C"
