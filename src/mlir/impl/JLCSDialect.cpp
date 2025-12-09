// JLCSDialect.cpp - JLCS dialect registration
#include "JLCSDialect.h"
#include "JLCSOps.h"
#include "JLCSTypes.h"

#include "JLCSDialect.cpp.inc"

using namespace mlir;
using namespace mlir::jlcs;

void JLCSDialect::initialize()
{
    // Register operations
    addOperations<
#define GET_OP_LIST
#include "JLCSOps.cpp.inc"
        >();

    // Register types
    addTypes<
#define GET_TYPEDEF_LIST
#include "JLCSTypes.cpp.inc"
        >();
}
