// AOTCompiler.cpp - Emit LLVM IR text from MLIR modules
//
// Two-step AOT strategy to avoid Julia LLVM 18 / system LLVM 21 symbol conflicts:
//   1. jlcs_emit_llvmir() — translates MLIR (LLVM dialect) → LLVM IR text file (in-process)
//   2. Julia shells out to `llc` to compile .ll → .o (out-of-process, no conflict)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/CAPI/IR.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

extern "C" {

/// Translate an MLIR module (already lowered to LLVM dialect) to LLVM IR text
/// and write it to the given file path.
/// Returns true on success.
bool jlcs_emit_llvmir(MlirModule module, const char* outputPath) {
    mlir::ModuleOp modOp = unwrap(module);
    llvm::LLVMContext llvmContext;

    auto llvmModule = mlir::translateModuleToLLVMIR(modOp, llvmContext);
    if (!llvmModule) {
        return false;
    }

    std::error_code ec;
    llvm::raw_fd_ostream dest(outputPath, ec);
    if (ec) {
        return false;
    }

    llvmModule->print(dest, nullptr);
    dest.flush();
    return true;
}

}
