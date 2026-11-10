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

    // Test seams: the call shape the lowering's buildCallShape gives every
    // jlcs.ffe_call and jlcs.try_call in `module` under one convention's
    // rules, on any host, one line per call (format at CallShapeProbe). Each
    // works on a clone, so `module` is unchanged. test/test_sysv_abi_oracle.jl
    // diffs the SysV answer against LLVM's ABI lowering library;
    // test/test_aapcs64_abi_spec.jl diffs the AAPCS64 answer against clang's
    // lowering for aarch64-linux-gnu.
    std::string describeSysVCallShapes(ModuleOp module);
    std::string describeAAPCS64CallShapes(ModuleOp module);
} // namespace jlcs
} // namespace mlir

#endif // MLIR_DIALECT_JLCS_JLCSDIALECT_H
