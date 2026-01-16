// JLCSCAPIWrappers.cpp - C API wrappers for Julia
//
// This file provides MLIR C API functions using the C++ API internally

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir-c/IR.h"
#include "mlir-c/ExecutionEngine.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

// ENVIRONMENT-SPECIFIC FIXES
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

#include "JLCSDialect.h"

using namespace mlir;

// --- Helper Functions ---

// helper to attach host data layout
static void attachHostDataLayout(mlir::ModuleOp module) {
    auto triple = llvm::sys::getProcessTriple();
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        llvm::errs() << "Warning: Failed to lookup target for data layout: " << error << "\n";
        return;
    }

    auto machine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(triple, "generic", "", {}, std::nullopt));

    const llvm::DataLayout &dl = machine->createDataLayout();
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(module.getContext(), dl.getStringRepresentation()));
}

extern "C" {

    // --- Dialect & Context Management ---

    void registerJLCSDialect(MlirContext context) {
        MLIRContext *ctx = unwrap(context);
        ctx->loadDialect<jlcs::JLCSDialect>();
        ctx->loadDialect<func::FuncDialect>();
        ctx->loadDialect<arith::ArithDialect>();
        ctx->loadDialect<LLVM::LLVMDialect>();
        // Important: Load translation for LLVM IR lowering
        mlir::DialectRegistry registry;
        mlir::registerAllToLLVMIRTranslations(registry);
        ctx->appendDialectRegistry(registry);
    }

    MlirContext mlirContextCreate() {
        auto *ctx = new MLIRContext();
        return wrap(ctx);
    }

    void mlirContextDestroy(MlirContext context) {
        delete unwrap(context);
    }

    // --- Module Management ---

    MlirLocation mlirLocationUnknownGet(MlirContext context) {
        MLIRContext *ctx = unwrap(context);
        Location loc = UnknownLoc::get(ctx);
        return wrap(loc);
    }

    MlirModule mlirModuleCreateEmpty(MlirLocation location) {
        auto mod = ModuleOp::create(unwrap(location));
        return wrap(mod);
    }

    MlirModule jlcsModuleCreateParse(MlirContext context, const char *moduleStr) {
        MLIRContext *ctx = unwrap(context);
        llvm::StringRef source(moduleStr);
        OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(source, ctx);
        if (!mod) {
            return {nullptr};
        }
        return wrap(mod.release());
    }

    MlirOperation mlirModuleGetOperation(MlirModule module) {
        return wrap(unwrap(module).getOperation());
    }

    void mlirOperationDump(MlirOperation op) {
        unwrap(op)->dump();
    }

    // --- JIT Execution Engine ---

    MlirExecutionEngine jlcs_create_jit(MlirModule module, int optLevel, bool dumpObject) {
        // 1. Initialize LLVM Native Targets (Mandatory for JIT)
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter(); 
        llvm::InitializeNativeTargetAsmParser();

        // Unwrap directly to ModuleOp
        mlir::ModuleOp modOp = unwrap(module);

        // 2. Attach Data Layout
        attachHostDataLayout(modOp);

        // 3. Configure JIT Options
        mlir::ExecutionEngineOptions options;
        options.transformer = [optLevel](llvm::Module *m) {
            return llvm::Error::success(); 
        };
        options.jitCodeGenOptLevel = (llvm::CodeGenOptLevel)optLevel;

        // 4. Create Engine
        auto engineOrError = mlir::ExecutionEngine::create(modOp, options);

        if (!engineOrError) {
            llvm::errs() << "Failed to create ExecutionEngine: " << engineOrError.takeError() << "\n";
            return {nullptr};
        }

        return wrap(engineOrError->release());
    }

    void jlcs_destroy_jit(MlirExecutionEngine jit) {
        mlirExecutionEngineDestroy(jit);
    }

    void jlcs_jit_register_symbol(MlirExecutionEngine jit, const char *name, void *addr) {
        llvm::sys::DynamicLibrary::AddSymbol(name, addr);
    }

    void *jlcs_jit_lookup(MlirExecutionEngine jit, const char *name) {
        MlirStringRef nameRef = mlirStringRefCreateFromCString(name);
        return mlirExecutionEngineLookup(jit, nameRef);
    }

    bool jlcs_jit_invoke(MlirExecutionEngine jit, const char *name, void **args) {
        MlirStringRef funcName = mlirStringRefCreateFromCString(name);
        MlirLogicalResult res = mlirExecutionEngineInvokePacked(jit, funcName, args);
        return mlirLogicalResultIsSuccess(res);
    }

}
