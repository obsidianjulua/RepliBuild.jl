// mlir_wrapper.c - Minimal C wrapper to expose MLIR C API for RepliBuild discovery
// This file exists purely for RepliBuild to discover MLIR C API types and functions

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/BuiltinAttributes.h"

// No implementation needed - RepliBuild will parse the headers and generate bindings
// The actual MLIR C API functions are in libMLIR-C.so (or similar)

// Dummy function to force header inclusion
void mlir_api_reference(void) {
    // This function is never called, just forces the compiler to see the headers
}
