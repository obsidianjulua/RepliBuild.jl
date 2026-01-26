//===- JLCSDialect.h - JLCS dialect -----------------------------*- C++ -*-===//
//
// Julia C-Struct Dialect for FFI type representation
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_JLCS_JLCSDIALECT_H
#define MLIR_DIALECT_JLCS_JLCSDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

#include "JLCSDialect.h.inc"

namespace mlir {
namespace jlcs {
    std::unique_ptr<Pass> createLowerJLCSToLLVMPass();
} // namespace jlcs
} // namespace mlir

#endif // MLIR_DIALECT_JLCS_JLCSDIALECT_H
