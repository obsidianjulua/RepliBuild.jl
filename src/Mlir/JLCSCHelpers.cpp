// JLCSCHelpers.cpp - Simple C wrapper around C++ MLIR API for Julia

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir-c/IR.h"
#include "mlir-c/ExecutionEngine.h"
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

//===----------------------------------------------------------------------===//
// JIT Execution Engine (passthrough to MLIR C API)
//===----------------------------------------------------------------------===//

// Create JIT execution engine
// Just forwards to mlirExecutionEngineCreate - no interference with MLIR
MlirExecutionEngine jlcs_create_jit(MlirModule module, int optLevel, bool enableObjectDump) {
    return mlirExecutionEngineCreate(module, optLevel, 0, nullptr, enableObjectDump);
}

// Destroy JIT
void jlcs_destroy_jit(MlirExecutionEngine jit) {
    mlirExecutionEngineDestroy(jit);
}

// Lookup function pointer by name
void* jlcs_jit_lookup(MlirExecutionEngine jit, const char* name) {
    MlirStringRef nameRef = mlirStringRefCreateFromCString(name);
    return mlirExecutionEngineLookup(jit, nameRef);
}

// Invoke function with packed arguments
bool jlcs_jit_invoke(MlirExecutionEngine jit, const char* name, void** args) {
    MlirStringRef nameRef = mlirStringRefCreateFromCString(name);
    return mlirLogicalResultIsSuccess(mlirExecutionEngineInvokePacked(jit, nameRef, args));
}

// Register external symbol (for linking C++ functions)
void jlcs_jit_register_symbol(MlirExecutionEngine jit, const char* name, void* ptr) {
    MlirStringRef nameRef = mlirStringRefCreateFromCString(name);
    mlirExecutionEngineRegisterSymbol(jit, nameRef, ptr);
}

// Dump compiled code to object file
void jlcs_jit_dump_to_object(MlirExecutionEngine jit, const char* filename) {
    MlirStringRef filenameRef = mlirStringRefCreateFromCString(filename);
    mlirExecutionEngineDumpToObjectFile(jit, filenameRef);
}

} // extern "C"
