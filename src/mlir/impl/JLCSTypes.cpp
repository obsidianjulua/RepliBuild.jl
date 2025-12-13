//===- JLCSTypes.cpp - JLCS dialect type implementations ------------------===//
//
// Minimal wrapper to include TableGen-generated type classes.
//
//===----------------------------------------------------------------------===//

#include "JLCSTypes.h"
#include "JLCSDialect.h"

using namespace mlir;
using namespace mlir::jlcs;

// Bring in the TableGen generated type class definitions.
#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.cpp.inc"
