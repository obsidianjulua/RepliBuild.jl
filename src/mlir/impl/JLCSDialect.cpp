//===- JLCSDialect.cpp - JLCS dialect implementation ---------------------===//

#include "JLCSDialect.h"
#include "JLCSOps.h"
#include "JLCSTypes.h"
#include "mlir-c/IR.h"  // For MlirContext
#include "mlir/CAPI/IR.h"  // For unwrap

using namespace mlir;
using namespace mlir::jlcs;

// Include the generated dialect definitions
#include "JLCSDialect.cpp.inc"

void JLCSDialect::initialize()
{
    // Register generated types
    addTypes<
#define GET_TYPEDEF_LIST
#include "JLCSTypes.cpp.inc"
        >();

    // Register generated ops
    addOperations<
#define GET_OP_LIST
#include "JLCSOps.cpp.inc"
        >();
}

// LLVM 21: parseType/printType are auto-generated when useDefaultTypePrinterParser = 1
// No need to implement them manually

// C API: exported registration helper is in JLCSCAPIWrappers.cpp
