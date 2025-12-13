//===- JLCSTypes.cpp - JLCS dialect type implementations ------------------===//
//
// Type implementation methods - Storage classes manually defined in header
//
//===----------------------------------------------------------------------===//

#include "JLCSTypes.h"
#include "JLCSDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::jlcs;

// Include generated type method implementations
// Note: Storage classes are already defined in JLCSTypes.h,
// so redefinitions in .cpp.inc will be ignored by the linker
#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.cpp.inc"
