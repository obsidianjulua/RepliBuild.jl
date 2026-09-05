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
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/ExecutionEngine.h"

#include "mlir/Parser/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

// ENVIRONMENT-SPECIFIC FIXES
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

#include "JLCSDialect.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

using namespace mlir;

// =============================================================================
// Thread-local exception buffer for C++ → Julia exception propagation
// =============================================================================

static thread_local char jlcs_exception_buffer[1024] = {0};
static thread_local bool jlcs_has_exception = false;

// =============================================================================
// Diagnostic capture
//
// MLIR reports everything it knows about a failure through its DiagnosticEngine
// — the op that is wrong, the type that will not translate, and a
// FileLineColLoc naming the buffer and line. With no handler registered, that
// all goes to the default handler, which prints to stderr and returns. Julia
// then raised "Failed to parse MLIR module" and the one thing that would have
// explained it was on a stream nothing was reading, in a wrapper load that
// often is not even attached to a terminal.
//
// The locations are the reason this is worth capturing rather than merely
// silencing: parses are named after the content-keyed .mlir that
// jit_source_path already writes, so a captured diagnostic points at the exact
// file and line that gdb opens when you break in the resulting thunk. The error
// text and the debugger agree.
//
// Handlers are registered per-context and consulted innermost-first, and
// returning success() marks the diagnostic handled so the default handler does
// NOT also print it. That is deliberate: one report, delivered to the caller
// that can act on it, instead of two — one of them into the void.
// =============================================================================

static thread_local std::string jlcs_diagnostics;

namespace {

// "error: /path/jlcs_ab12.mlir:418:7: message", plus any attached notes
// indented beneath it. Severity is spelled out because a captured warning and a
// captured error look identical once they are both just text in a buffer.
void appendDiagnostic(std::string &out, mlir::Diagnostic &diag) {
    llvm::raw_string_ostream os(out);
    if (!out.empty()) {
        os << "\n";
    }
    switch (diag.getSeverity()) {
    case mlir::DiagnosticSeverity::Note:          os << "note: ";      break;
    case mlir::DiagnosticSeverity::Warning:       os << "warning: ";   break;
    case mlir::DiagnosticSeverity::Error:         os << "error: ";     break;
    case mlir::DiagnosticSeverity::Remark:        os << "remark: ";    break;
    }
    os << diag.getLocation() << ": " << diag.str();
    for (mlir::Diagnostic &note : diag.getNotes()) {
        os << "\n  note: " << note.getLocation() << ": " << note.str();
    }
    os.flush();
}

// Append a message with no Diagnostic behind it — for the failures this file
// detects itself (a refused pre-flight, an llvm::Error from the engine) rather
// than ones MLIR reports through its own engine. Same buffer, so the caller has
// one place to look regardless of which layer objected.
void appendPlain(std::string &out, llvm::StringRef msg) {
    if (!out.empty()) {
        out += "\n";
    }
    out += "error: ";
    out += msg.str();
}

// RAII capture for the duration of one fallible operation.
//
// Scoped rather than registered once at context creation because the buffer is
// per-thread and per-attempt: a caller reads it immediately after a failure,
// and a handler outliving the call would accumulate diagnostics from unrelated
// work into the next failure's report.
class DiagnosticCapture {
public:
    explicit DiagnosticCapture(mlir::MLIRContext *ctx)
        : handler(ctx, [](mlir::Diagnostic &diag) {
              appendDiagnostic(jlcs_diagnostics, diag);
              return mlir::success();
          }) {
        jlcs_diagnostics.clear();
    }

private:
    mlir::ScopedDiagnosticHandler handler;
};

} // namespace

// --- Helper Functions ---

// helper to attach host data layout
// Attach the host's data layout to the module. Returns false if it could not be
// determined, having said why in the diagnostic buffer.
//
// Two things changed here, and the second is the reason for the first.
//
// `createTargetMachine` can return null, and the result was dereferenced
// unchecked one line later — a null deref is an uncatchable SIGSEGV that takes
// the host process with it, which is the exact failure mode
// moduleTypesAreLLVMCompatible was written to prevent thirty lines below. The
// `target` lookup above it was already checked; this was not.
//
// The warn-and-continue was worse than the crash, though. This attribute is
// what tells the LLVM conversion passes how wide a pointer is and where struct
// fields land. Without it they fall back to a default layout and carry on
// happily — so a wrapper whose entire job is to match the host ABI would lower,
// JIT, run, and return quietly wrong answers, having printed a warning to a
// stderr nobody reads. Failing here is the only honest option: a guess about
// the ABI is not a degraded mode of a thing that exists to get the ABI right.
static bool attachHostDataLayout(mlir::ModuleOp module) {
    llvm::Triple triple(llvm::sys::getProcessTriple());
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        appendPlain(jlcs_diagnostics,
                    "no LLVM target registered for host triple " + triple.str() +
                        ": " + error +
                        " (InitializeNativeTarget may not have run)");
        return false;
    }

    auto machine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(triple, "generic", "", {}, std::nullopt));
    if (!machine) {
        appendPlain(jlcs_diagnostics,
                    "could not create a TargetMachine for host triple " +
                        triple.str() + "; host data layout is unavailable");
        return false;
    }

    const llvm::DataLayout &dl = machine->createDataLayout();
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(module.getContext(), dl.getStringRepresentation()));
    return true;
}

// Pre-flight: refuse modules containing types that cannot be translated to
// LLVM IR. translateModuleToLLVMIR does not diagnose foreign types — a stray
// non-LLVM type (e.g. a !jlcs.* type that survived lowering inside an
// !llvm.struct body) sends it through a garbage interface pointer, an
// uncatchable SIGSEGV that kills the whole host process (observed live:
// PtrLikeTypeInterface::getMemorySpace during pugixml wrapper load,
// 2026-07-18). Failing here instead returns null to Julia, which raises a
// catchable error and degrades to Tier-2-disabled.
static bool moduleTypesAreLLVMCompatible(mlir::ModuleOp modOp) {
    bool ok = true;
    modOp->walk([&](mlir::Operation *op) {
        if (op == modOp.getOperation())
            return;
        auto check = [&](mlir::Type t) {
            if (!t)
                return;
            if (!mlir::LLVM::isCompatibleType(t)) {
                op->emitError("type is not translatable to LLVM IR: ") << t;
                ok = false;
            }
        };
        for (mlir::Type t : op->getOperandTypes())
            check(t);
        for (mlir::Type t : op->getResultTypes())
            check(t);
        for (mlir::Region &r : op->getRegions())
            for (mlir::Block &b : r)
                for (mlir::Type t : b.getArgumentTypes())
                    check(t);
        if (auto fn = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(op))
            check(fn.getFunctionType());
    });
    return ok;
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

    // sourceName names the parsed buffer instead of leaving MLIR's default "-"
    // for in-memory text. The parser stamps a FileLineColLoc carrying that name
    // onto every op, createDIScopeForLLVMFuncOpPass turns it into a DIFile, and
    // the emitted DWARF then points gdb at a path it can actually open — so
    // stepping a JIT'd thunk steps the generated MLIR. NULL/empty keeps the old
    // unnamed behaviour, which only costs the source view.
    //
    // ABI note: an older two-parameter libJLCS called with three arguments is
    // harmless under x86-64 SysV (the extra register is ignored), so a stale
    // symlinked dialect build degrades to unnamed parsing rather than
    // misbehaving — the worktree case.
    MlirModule jlcsModuleCreateParse(MlirContext context, const char *moduleStr,
                                     const char *sourceName) {
        MLIRContext *ctx = unwrap(context);
        DiagnosticCapture capture(ctx);
        llvm::StringRef source(moduleStr);
        llvm::StringRef name(sourceName ? sourceName : "");
        OwningOpRef<ModuleOp> mod =
            parseSourceString<ModuleOp>(source, ParserConfig(ctx), name);
        if (!mod) {
            return {nullptr};
        }
        return wrap(mod.release());
    }

    // Diagnostics captured during the most recent fallible call ON THIS THREAD,
    // or "" if it had nothing to say. Empty is a real answer, not an error: a
    // clean parse emits nothing.
    //
    // The pointer is into a thread_local std::string that the next capture
    // clears, so the caller must copy before calling back in — which is what
    // Julia's unsafe_string does.
    const char *jlcs_get_diagnostics() {
        return jlcs_diagnostics.c_str();
    }

    void jlcs_clear_diagnostics() {
        jlcs_diagnostics.clear();
    }

    MlirModule jlcs_module_clone(MlirModule module) {
        mlir::ModuleOp mod = unwrap(module);
        mlir::ModuleOp cloned = llvm::cast<mlir::ModuleOp>(mod->clone());
        return wrap(cloned);
    }

    MlirOperation mlirModuleGetOperation(MlirModule module) {
        return wrap(unwrap(module).getOperation());
    }

    void mlirOperationDump(MlirOperation op) {
        unwrap(op)->dump();
    }

    // --- Introspection ---

    MlirOperation jlcs_module_get_function(MlirModule module, const char *name) {
        mlir::ModuleOp mod = unwrap(module);
        mlir::func::FuncOp func = mod.lookupSymbol<mlir::func::FuncOp>(name);
        if (!func) return {nullptr};
        return wrap(func.getOperation());
    }

    MlirType jlcs_function_get_type(MlirOperation op) {
        auto func = llvm::dyn_cast<mlir::func::FuncOp>(unwrap(op));
        if (!func) return {nullptr};
        return wrap(func.getFunctionType());
    }

    intptr_t jlcs_function_type_get_num_inputs(MlirType type) {
        return mlirFunctionTypeGetNumInputs(type);
    }

    MlirType jlcs_function_type_get_input(MlirType type, intptr_t pos) {
        return mlirFunctionTypeGetInput(type, pos);
    }

    bool jlcs_type_is_integer(MlirType type) {
        return mlirTypeIsAInteger(type);
    }

    unsigned jlcs_integer_type_get_width(MlirType type) {
        return mlirIntegerTypeGetWidth(type);
    }

    bool jlcs_type_is_f32(MlirType type) {
        return mlirTypeIsAF32(type);
    }

    bool jlcs_type_is_f64(MlirType type) {
        return mlirTypeIsAF64(type);
    }

    // --- Transformations ---

    bool jlcs_lower_to_llvm(MlirModule module) {
        mlir::ModuleOp mod = unwrap(module);
        DiagnosticCapture capture(mod.getContext());
        mlir::PassManager pm(mod.getContext());
        // Disable inter-pass verification: jlcs-lower-to-llvm emits llvm.call ops
        // that reference func.func symbols which won't become llvm.func until the
        // next pass (ConvertFuncToLLVM). The final module is still fully valid.
        pm.enableVerifier(false);

        // Add JLCS custom lowering pass FIRST
        pm.addPass(mlir::jlcs::createLowerJLCSToLLVMPass());

        // Basic lowering pipeline
        // Convert Func -> LLVM
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        // Convert Arith -> LLVM
        pm.addPass(mlir::createArithToLLVMConversionPass());
        // Cleanup casts
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        // Materialize a DISubprogram on every llvm.func that lacks one. Must run
        // LAST: it operates on LLVM::LLVMFuncOp, which only exist after
        // ConvertFuncToLLVM above.
        //
        // Without this the emitted object carries no DWARF at all, so the JIT
        // event listeners registered in jlcs_create_jit_with_libs have nothing to
        // report: perf jitdumps contain zero JIT_CODE_DEBUG_INFO records and gdb
        // gets symbol names but no function scopes. Defaults to LineTablesOnly,
        // which is what makes backtraces useful without the size cost of Full.
        //
        // The line table points somewhere REAL for anything that came through
        // jlcsModuleCreateParse: the parser stamps a FileLineColLoc naming the
        // buffer, this pass turns it into the DIFile, and the caller writes that
        // buffer to the named path — so gdb's `list` shows the generated MLIR.
        // Only the programmatic path (jlcsModuleCreate, UnknownLoc above) gets a
        // synthetic file/line, and nothing currently ships thunks that way.
        pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

        bool ok = mlir::succeeded(pm.run(mod));

        // Post-pipeline fixup: ensure any llvm.func containing llvm.invoke
        // has the host's C++ personality set. This must happen after ALL
        // passes (including ConvertFuncToLLVM) have completed.
        //
        // The name is target-dependent and must agree with JLCSPasses.cpp's
        // kCxxPersonality: Itanium unwinding calls it `__gxx_personality_v0`,
        // mingw-w64 x86-64 unwinds with SEH and calls it
        // `__gxx_personality_seh0`, and the Windows C++ runtime exports only
        // the latter — so emitting v0 there links with one undefined symbol.
        // Two copies of one fact, so they are spelled identically and both say
        // so; if you change one, change the other.
#if defined(_WIN32)
        static constexpr const char *kCxxPersonality = "__gxx_personality_seh0";
#else
        static constexpr const char *kCxxPersonality = "__gxx_personality_v0";
#endif
        mod.walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
            bool hasInvoke = false;
            funcOp.walk([&](mlir::LLVM::InvokeOp) { hasInvoke = true; });
            if (hasInvoke && !funcOp.getPersonalityAttr()) {
                // Ensure the personality routine is declared
                if (!mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(kCxxPersonality)) {
                    mlir::OpBuilder builder(mod.getContext());
                    builder.setInsertionPointToStart(mod.getBody());
                    auto i32Type = builder.getI32Type();
                    auto personalityFnType = mlir::LLVM::LLVMFunctionType::get(i32Type, {}, true);
                    mlir::LLVM::LLVMFuncOp::create(builder, mod.getLoc(), kCxxPersonality, personalityFnType);
                }
                auto personalityRef = mlir::FlatSymbolRefAttr::get(
                    mod.getContext(), kCxxPersonality);
                funcOp.setPersonalityAttr(personalityRef);
            }
        });

        return ok;
    }

    // --- JIT Execution Engine ---

    // Forward declaration
    MlirExecutionEngine jlcs_create_jit_with_libs(MlirModule module, int optLevel, bool dumpObject,
                                                    const char **sharedLibPaths, int numLibs);

    MlirExecutionEngine jlcs_create_jit(MlirModule module, int optLevel, bool dumpObject) {
        return jlcs_create_jit_with_libs(module, optLevel, dumpObject, nullptr, 0);
    }

    MlirExecutionEngine jlcs_create_jit_with_libs(MlirModule module, int optLevel, bool dumpObject,
                                                    const char **sharedLibPaths, int numLibs) {
        // 1. Initialize LLVM Native Targets (Mandatory for JIT)
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        // Unwrap directly to ModuleOp
        mlir::ModuleOp modOp = unwrap(module);
        DiagnosticCapture capture(modOp.getContext());

        // 1b. Pre-flight type check — see moduleTypesAreLLVMCompatible. A bad
        // type must fail here (null return, catchable in Julia), not SIGSEGV
        // inside translateModuleToLLVMIR.
        //
        // The guard's own emitError names the offending op and type, which is
        // the entire reason it emits one — it used to go to stderr and be lost,
        // leaving the refusal explained by nothing. It is now captured with the
        // rest and reaches Julia.
        if (!moduleTypesAreLLVMCompatible(modOp)) {
            appendPlain(jlcs_diagnostics,
                        "module contains types that cannot be translated to "
                        "LLVM IR; refusing to JIT");
            return {nullptr};
        }

        // 2. Attach Data Layout — refuse to JIT without it, see the function.
        if (!attachHostDataLayout(modOp)) {
            return {nullptr};
        }

        // 3. Configure JIT Options
        mlir::ExecutionEngineOptions options;
        options.transformer = [optLevel](llvm::Module *m) {
            return llvm::Error::success();
        };
        options.jitCodeGenOptLevel = (llvm::CodeGenOptLevel)optLevel;

        // JIT introspection listeners. MLIR turns BOTH on by default
        // (ExecutionEngineOptions in mlir/ExecutionEngine/ExecutionEngine.h) —
        // they are not registered by anything in this file, they arrive with
        // MLIRExecutionEngine, which CMakeLists links --whole-archive.
        //
        //   GDB  — free, no filesystem side effects, and it is what lets gdb
        //          resolve a thunk by its mangled name and step the generated
        //          MLIR. Always on.
        //   perf — writes a jitdump per PROCESS to $JITDUMPDIR/.debug/jit
        //          (LLVM hardcodes the ".debug/jit" suffix; without JITDUMPDIR
        //          it lands in $HOME). Nothing rotates or expires it: 718
        //          directories / 164MB had accumulated unnoticed by 2026-08-08,
        //          because the default is on and nobody knew. Opt-in now.
        options.enableGDBNotificationListener = true;
        options.enablePerfNotificationListener =
            (std::getenv("REPLIBUILD_JIT_PROFILE") != nullptr);

        // Object cache. `dumpObject` has been a parameter of this function — and
        // of Julia's create_jit — since the JIT was written, threaded all the
        // way down and then dropped on the floor here, so the knob read as
        // supported and did nothing. It is the precondition for
        // jlcs_dump_object below: MLIR only retains the emitted object if the
        // cache exists at engine-creation time, and it cannot be turned on
        // afterwards. Off by default because the cache holds every emitted
        // object for the engine's lifetime.
        options.enableObjectDump = dumpObject;

        // 4. Register shared libraries for symbol resolution
        SmallVector<StringRef, 4> libPaths;
        for (int i = 0; i < numLibs; i++) {
            if (sharedLibPaths[i]) {
                libPaths.push_back(sharedLibPaths[i]);
            }
        }
        options.sharedLibPaths = libPaths;

        // 5. Create Engine
        auto engineOrError = mlir::ExecutionEngine::create(modOp, options);

        if (!engineOrError) {
            // takeError() consumes it — an llvm::Error that is neither consumed
            // nor handled aborts the process in a debug LLVM, so this must
            // happen exactly once and the text has to be kept here.
            appendPlain(jlcs_diagnostics,
                        "failed to create ExecutionEngine: " +
                            llvm::toString(engineOrError.takeError()));
            return {nullptr};
        }

        return wrap(engineOrError->release());
    }

    void jlcs_destroy_jit(MlirExecutionEngine jit) {
        mlirExecutionEngineDestroy(jit);
    }

    // Register `name` -> `addr` for symbol resolution.
    //
    // A non-null engine registers ON that engine, which is what this signature
    // has always promised and never did: the parameter was accepted and
    // dropped, so Julia's `register_symbol(jit, ...)` — which takes an engine
    // and documents per-engine behaviour — fell through to the process-global
    // table exactly like `register_symbol_global`. Two wrappers defining the
    // same mangled name then shared one slot silently, last writer winning, and
    // the loser's thunks dispatched into the other library's code.
    //
    // A null engine still means the global table, and that is not a leftover.
    // ORC resolves a symbol when it materialises the code referencing it, so
    // anything the engine must see during ExecutionEngine::create has to be
    // registered before the engine exists — which is precisely when
    // initialize_global_jit registers its dispatch_ and EH helpers. Passing
    // null is how a caller says "there is no engine yet"; it is the honest
    // spelling of what the old code did unconditionally.
    void jlcs_jit_register_symbol(MlirExecutionEngine jit, const char *name, void *addr) {
        if (unwrap(jit) != nullptr) {
            mlirExecutionEngineRegisterSymbol(
                jit, mlirStringRefCreateFromCString(name), addr);
            return;
        }
        llvm::sys::DynamicLibrary::AddSymbol(name, addr);
    }

    // Write the JIT's emitted object file to `path`. Requires the engine to
    // have been created with dumpObject=true; the cache cannot be added later.
    //
    // Returns success rather than void because MLIR's dumpToObjectFile REPORTS
    // a disabled cache by printing to stderr and returning — indistinguishable
    // from success at the call site, and stderr is exactly where a wrapper's
    // diagnostics get lost. Existence + non-empty is checked here so the caller
    // gets a value it can fail on.
    bool jlcs_dump_object(MlirExecutionEngine jit, const char *path) {
        if (!path || !*path) {
            return false;
        }
        mlir::ExecutionEngine *engine = unwrap(jit);
        if (!engine) {
            return false;
        }
        engine->dumpToObjectFile(path);
        uint64_t size = 0;
        if (llvm::sys::fs::file_size(path, size)) {
            return false;   // non-zero error code: absent or unreadable
        }
        return size > 0;
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

    // --- Self-Test: one-shot JIT diagnostic ---
    // Does parse → lower → ExecutionEngine → invokePacked entirely in C++.
    // If this crashes when called from Julia, the Julia runtime environment
    // (signals, mmap, etc.) is interfering with ORC JIT codegen.
    // If it succeeds, the multi-ccall orchestration is the problem.
    int jlcs_selftest_jit_add(int32_t a, int32_t b, int32_t *result) {
        // 1. Init targets (idempotent)
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        // 2. Fresh context + dialects
        mlir::MLIRContext ctx;
        ctx.loadDialect<mlir::func::FuncDialect,
                        mlir::arith::ArithDialect,
                        mlir::LLVM::LLVMDialect>();
        mlir::registerBuiltinDialectTranslation(ctx);
        mlir::registerLLVMDialectTranslation(ctx);

        // 3. Parse trivial IR
        const char *ir = R"MLIR(
            module {
              func.func @add(%a: i32, %b: i32) -> i32
                  attributes {llvm.emit_c_interface} {
                %r = arith.addi %a, %b : i32
                return %r : i32
              }
            }
        )MLIR";
        auto mod = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
        if (!mod) { llvm::errs() << "[selftest] parse failed\n"; return 1; }

        // 4. Lower to LLVM dialect
        {
            mlir::PassManager pm(&ctx);
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            if (mlir::failed(pm.run(*mod))) {
                llvm::errs() << "[selftest] lowering failed\n";
                return 2;
            }
        }

        // 5. Attach data layout
        if (!attachHostDataLayout(*mod)) {
            llvm::errs() << "[selftest] host data layout unavailable: "
                         << jlcs_diagnostics << "\n";
            return 3;
        }

        // 6. Create ExecutionEngine
        mlir::ExecutionEngineOptions opts;
        opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
        opts.transformer = [](llvm::Module *) { return llvm::Error::success(); };

        auto maybeEngine = mlir::ExecutionEngine::create(*mod, opts);
        if (!maybeEngine) {
            llvm::errs() << "[selftest] ExecutionEngine::create failed: "
                         << maybeEngine.takeError() << "\n";
            return 3;
        }
        auto engine = std::move(*maybeEngine);

        // 7. invokePacked — the crash site
        void *argPtrs[] = { &a, &b, result };
        if (auto err = engine->invokePacked("add", argPtrs)) {
            llvm::errs() << "[selftest] invokePacked error: " << err << "\n";
            return 4;
        }

        return 0;  // success
    }

    // --- Exception Handling C API ---

    void jlcs_set_pending_exception(const char* msg) {
        if (msg) {
            std::strncpy(jlcs_exception_buffer, msg, sizeof(jlcs_exception_buffer) - 1);
            jlcs_exception_buffer[sizeof(jlcs_exception_buffer) - 1] = '\0';
        } else {
            std::strncpy(jlcs_exception_buffer, "unknown C++ exception",
                         sizeof(jlcs_exception_buffer) - 1);
            jlcs_exception_buffer[sizeof(jlcs_exception_buffer) - 1] = '\0';
        }
        jlcs_has_exception = true;
    }

    bool jlcs_has_pending_exception() {
        return jlcs_has_exception;
    }

    const char* jlcs_get_pending_exception() {
        return jlcs_exception_buffer;
    }

    void jlcs_clear_pending_exception() {
        jlcs_has_exception = false;
        jlcs_exception_buffer[0] = '\0';
    }

    /// Called from JIT'd landing pad code to extract the exception message
    /// from the caught C++ exception and store it in the thread-local buffer.
    const char* jlcs_catch_current_exception() {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& e) {
            jlcs_set_pending_exception(e.what());
            return jlcs_exception_buffer;
        } catch (...) {
            jlcs_set_pending_exception("unknown C++ exception");
            return jlcs_exception_buffer;
        }
    }

}
