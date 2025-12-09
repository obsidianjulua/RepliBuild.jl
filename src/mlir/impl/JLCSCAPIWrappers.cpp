// JLCSCAPIWrappers.cpp - C API wrappers for Julia
//
// This file provides MLIR C API functions using the C++ API internally

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir-c/IR.h"
#include "JLCSDialect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

extern "C" {

// Dialect registration
void registerJLCSDialect(MlirContext context) {
    MLIRContext *ctx = unwrap(context);
    ctx->getOrLoadDialect<jlcs::JLCSDialect>();
}

// Context management - use C++ API directly
MlirContext mlirContextCreate() {
    auto *ctx = new MLIRContext();
    return wrap(ctx);
}

void mlirContextDestroy(MlirContext context) {
    delete unwrap(context);
}

// Location management
MlirLocation mlirLocationUnknownGet(MlirContext context) {
    MLIRContext *ctx = unwrap(context);
    Location loc = UnknownLoc::get(ctx);
    return wrap(loc);
}

// Module management
MlirModule mlirModuleCreateEmpty(MlirLocation location) {
    auto mod = ModuleOp::create(unwrap(location));
    return wrap(mod);
}

MlirOperation mlirModuleGetOperation(MlirModule module) {
    return wrap(unwrap(module).getOperation());
}

// Operation management
void mlirOperationDump(MlirOperation op) {
    unwrap(op)->dump();
}

} // extern "C"
