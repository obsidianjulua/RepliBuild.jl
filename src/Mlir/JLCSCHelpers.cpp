// JLCSCHelpers.cpp - Simple C wrapper around C++ MLIR API for Julia

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"

using namespace mlir;

extern "C" {

// Simple context creation
MlirContext jlcs_create_context() {
    auto *ctx = new MLIRContext();
    return wrap(ctx);
}

// Simple context destruction
void jlcs_destroy_context(MlirContext context) {
    delete unwrap(context);
}

// Create empty module
MlirModule jlcs_create_module(MlirContext context) {
    MLIRContext *ctx = unwrap(context);
    Location loc = UnknownLoc::get(ctx);
    ModuleOp mod = ModuleOp::create(loc);
    return wrap(mod);
}

// Print module
void jlcs_print_module(MlirModule mlir_module) {
    ModuleOp mod = unwrap(mlir_module);
    mod.print(llvm::outs());
}

} // extern "C"
