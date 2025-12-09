//===- JLCSOps.cpp - JLCS dialect operations -----------------------------===//
//
// Operation implementations and verifiers for JLCS dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "JLCSDialect.h"
#include "JLCSOps.h"
#include "JLCSTypes.h"

using namespace mlir;
using namespace mlir::jlcs;

//===----------------------------------------------------------------------===//
// Operation implementations
//===----------------------------------------------------------------------===//

// Note: Operation verification is handled by traits and TableGen-generated code
// Custom verification can be added via Op::hasVerifier if needed

//===----------------------------------------------------------------------===//
// TableGen generated operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "JLCSOps.cpp.inc"
