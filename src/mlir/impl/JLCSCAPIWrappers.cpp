// JLCSCAPIWrappers.cpp - C API wrappers for Julia
//
// This file wraps MLIR C API functions that Julia needs, since the static
// MLIRCAPIIR library isn't available as a shared library on some systems.

#include "mlir-c/IR.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"

extern "C" {

// Context management
MlirContext mlirContextCreate() {
    return ::mlirContextCreate();
}

void mlirContextDestroy(MlirContext context) {
    ::mlirContextDestroy(context);
}

// Location management
MlirLocation mlirLocationUnknownGet(MlirContext context) {
    return ::mlirLocationUnknownGet(context);
}

// Module management
MlirModule mlirModuleCreateEmpty(MlirLocation location) {
    return ::mlirModuleCreateEmpty(location);
}

MlirOperation mlirModuleGetOperation(MlirModule module) {
    return ::mlirModuleGetOperation(module);
}

// Operation management
void mlirOperationDump(MlirOperation op) {
    ::mlirOperationDump(op);
}

} // extern "C"
