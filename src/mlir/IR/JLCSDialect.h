//===- JLCSDialect.h - JLCS dialect -----------------------------*- C++ -*-===//
//
// Julia C-Struct Dialect for FFI type representation
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_JLCS_JLCSDIALECT_H
#define MLIR_DIALECT_JLCS_JLCSDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

#include <string>

#include "JLCSDialect.h.inc"

namespace mlir {
class ModuleOp;
namespace jlcs {
    std::unique_ptr<Pass> createLowerJLCSToLLVMPass();

    // Test seam: the call shape the lowering's buildCallShape gives every
    // jlcs.ffe_call and jlcs.try_call in `module` under the x86-64 SysV rules,
    // on any host, one line per call (format at SysVCallShapeProbe). Works on
    // a clone, so `module` is unchanged. test/test_sysv_abi_oracle.jl diffs
    // it against LLVM's ABI lowering library.
    std::string describeSysVCallShapes(ModuleOp module);
} // namespace jlcs
} // namespace mlir

#endif // MLIR_DIALECT_JLCS_JLCSDIALECT_H
