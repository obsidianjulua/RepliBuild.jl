// JLCSDialect.cpp (sketch)
#include "JLCSDialect.h"

//
#include "JLCSDialect.cpp.inc"

//
// must be before dialect initialize()
#include "JLCSDialect.cpp.inc"

using namespace mlir;
using namespace mlir::jlir::jlcs;

void JLCSDialect::initialize()
{
    // Generated helpers add the ops/types declared in TableGen
    addOperations<
#define GET_OP_LIST
#include "JLCSOps.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "JLCSTypes.cpp.inc"
        >();

    // If you want explicit custom registration hooks:
    registerTypes();
    registerOps();
}
