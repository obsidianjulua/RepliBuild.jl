//===- JLCSTypes.h - JLCS dialect types -------------------------*- C++ -*-===//
//
// Type system for the JLCS dialect
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_JLCS_JLCSTYPES_H
#define MLIR_DIALECT_JLCS_JLCSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.h.inc"

#endif // MLIR_DIALECT_JLCS_JLCSTYPES_H
