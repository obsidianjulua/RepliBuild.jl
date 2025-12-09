//===- JLCSTypes.h - JLCS dialect types -------------------------*- C++ -*-===//
//
// Type system for the JLCS dialect
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_JLCS_JLCSTYPES_H
#define MLIR_DIALECT_JLCS_JLCSTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

// Forward declare detail namespace
namespace mlir {
namespace jlcs {
namespace detail {
  struct CStructTypeStorage;
} // namespace detail
} // namespace jlcs
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.h.inc"

#endif // MLIR_DIALECT_JLCS_JLCSTYPES_H
