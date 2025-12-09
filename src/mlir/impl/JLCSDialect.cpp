// JLCSDialect.cpp - JLCS dialect registration
#include "JLCSDialect.h"
#include "JLCSOps.h"
#include "JLCSTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::jlcs;

// IMPORTANT: Include type storage definitions BEFORE registering
// This makes the complete storage class visible for registration
#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.cpp.inc"

#include "JLCSDialect.cpp.inc"

void JLCSDialect::initialize()
{
    // Register operations
    addOperations<
#define GET_OP_LIST
#include "JLCSOps.cpp.inc"
        >();

    // Register types (storage already included above)
    addTypes<CStructType, ArrayViewType>();
}
