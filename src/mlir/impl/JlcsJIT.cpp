// JlcsJIT.cpp — dialect-owned ORC LLJIT.
//
// Why this file exists
// --------------------
// mlir::ExecutionEngine::create (LLVM 22.1.8) builds:
//
//   LLJITBuilder().setObjectLinkingLayerCreator([&] {
//     return RTDyldObjectLinkingLayer(session, [] {
//       return SectionMemoryManager(options.sectionMemoryMapper);
//     });
//   })
//
// Proven from the options struct: sectionMemoryMapper is
// llvm::SectionMemoryManager::MemoryMapper*, the RTDyld-shaped knob, and there
// is no ObjectLinkingLayer hook. RuntimeDyld, not JITLink.
//
// RuntimeDyldCOFFX86_64::finalizeLoad records every .pdata section and
// registerEHFrames hands it to MemMgr.registerEHFrames. The default
// implementation is RTDyldMemoryManager::registerEHFramesInProcess →
// __register_frame, which is the ELF DWARF mechanism.
//
// COFF x86-64 does not use .eh_frame. It needs RtlAddFunctionTable over the
// object's RUNTIME_FUNCTION entries. Feeding .pdata to __register_frame does
// two things, both wrong:
//
//   1. JIT'd frames never appear in the OS function table, so anything
//      unwinding through one (a C++ throw caught by try_call's landing pad)
//      gets libunwind: pc not in table, with a JIT-range PC.
//   2. __deregister_frame(.pdata) on engine teardown parses SEH bytes as
//      DWARF FDEs and poisons the process-global unwind forest. Subsequent
//      unwinds fail even for live JIT, which is why the crash wants "prior
//      in-parent JIT activity" and will not reproduce standalone: the first
//      engine's destructor is the poison, the next throw is the symptom.
//      That reads as a lifetime bug. It is the same missing-table bug, seen
//      from teardown.
//
// Why not JITLink
// ---------------
// JITLink/COFF_x86_64.h ships, and LLJITBuilder::setObjectLinkingLayerCreator
// would take an ObjectLinkingLayer. That does not register .pdata either —
// COFFPlatform does, and it needs the ORC runtime. We would still write a
// plugin that calls RtlAddFunctionTable. RuntimeDyld already delivers .pdata
// to registerEHFrames; overriding that is the one-line-shaped fix.
//
// The GDB JIT listener (enableGDBNotificationListener) is also
// RTDyld-only. JITLink's GDBJITDebugInfoRegistrationPlugin is MachO; switching
// linkers would drop source-level thunk debugging on ELF too.
//
// So: same RTDyld stack MLIR builds, dialect-owned so the memory manager can
// call RtlAddFunctionTable and must not call __register_frame.
//
// Catching is a second, separate problem. An in-JIT landing pad needs a
// Win64 EXCEPTION_ROUTINE at an RVA inside the JIT image. RuntimeDyld
// implements that as an ADDR32NB jmp stub; Windows then executes the stub
// as a handler and AVs into .xdata in a nested-exception loop. try_call /
// may_throw therefore lower to a plain call on Windows, and libJLCS
// (jlcs_guard_*, jlcs_jit_invoke) is the C++ frame that catches. .pdata
// is still required so the throw can *walk* the JIT frames to get there.

#include "impl/JlcsJIT.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

using llvm::Error;
using llvm::Expected;
using llvm::StringRef;
using llvm::orc::ExecutionSession;
using llvm::orc::MangleAndInterner;
using llvm::orc::RTDyldObjectLinkingLayer;
using llvm::orc::ThreadSafeModule;

namespace {

Error makeStringError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

std::string packedName(StringRef name) { return ("_mlir_" + name).str(); }

// Identical to mlir::ExecutionEngine's packFunctionArguments: every defined
// function `foo` gets a `void _mlir_foo(i8**)` wrapper. The C API's
// invokePacked prefixes `_mlir_ciface_` first, so the looked-up symbol is
// `_mlir__mlir_ciface_foo`. Julia's lookup() path does not use this; it
// binds `_mlir_ciface_foo` directly. Both have to keep working.
void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  llvm::DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration() || func.hasLocalLinkage())
      continue;
    if (interfaceFunctions.count(&func))
      continue;

    auto *newType = llvm::FunctionType::get(builder.getVoidTy(),
                                            builder.getPtrTy(),
                                            /*isVarArg=*/false);
    auto newName = packedName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc =
        llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto [index, arg] : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, index));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(builder.getPtrTy(), argPtrPtr);
      args.push_back(builder.CreateLoad(arg.getType(), argPtr));
    }

    llvm::Value *result = builder.CreateCall(&func, args);
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(builder.getPtrTy(), retPtrPtr);
      builder.CreateStore(result, retPtr);
    }
    builder.CreateRetVoid();
  }
}

#ifdef _WIN32

// Per-object memory manager. RTDyldObjectLinkingLayer constructs one of these
// per MemoryBuffer, so imageBase is the min load address of THIS object — the
// same value RuntimeDyldCOFFX86_64 uses for IMAGE_REL_AMD64_ADDR32NB, which
// is how .pdata's RVAs are written.
//
// ReserveAlloc keeps every section of the object inside one reservation so
// those RVAs fit in 32 bits (the ADDR32NB contract). MLIR only turns this on
// for AArch64; COFF needs it for the same "sections must be close" reason.
class Win64SEHMemoryManager : public llvm::SectionMemoryManager {
public:
  Win64SEHMemoryManager()
      : SectionMemoryManager(/*MM=*/nullptr, /*ReserveAlloc=*/true) {}

  ~Win64SEHMemoryManager() override { deregisterEHFrames(); }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    uint8_t *p = SectionMemoryManager::allocateCodeSection(Size, Alignment,
                                                           SectionID, SectionName);
    noteAlloc(p);
    return p;
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool isReadOnly) override {
    uint8_t *p = SectionMemoryManager::allocateDataSection(
        Size, Alignment, SectionID, SectionName, isReadOnly);
    noteAlloc(p);
    return p;
  }

  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                        size_t Size) override {
    (void)LoadAddr;
    // Addr is the relocated .pdata, not a DWARF .eh_frame. Do not call the
    // parent — that is __register_frame, and that is the poison.
    if (!Addr || imageBase == ~uint64_t(0) ||
        Size < sizeof(RUNTIME_FUNCTION))
      return;
    DWORD count = static_cast<DWORD>(Size / sizeof(RUNTIME_FUNCTION));
    auto *table = reinterpret_cast<PRUNTIME_FUNCTION>(Addr);
    if (!RtlAddFunctionTable(table, count, imageBase)) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        llvm::errs() << "JlcsJIT: RtlAddFunctionTable failed (GetLastError="
                     << GetLastError()
                     << "); JIT frames have no SEH unwind info\n";
      }
      return;
    }
    tables.push_back(table);
  }

  void deregisterEHFrames() override {
    for (PRUNTIME_FUNCTION t : tables)
      RtlDeleteFunctionTable(t);
    tables.clear();
  }

private:
  void noteAlloc(uint8_t *p) {
    if (p) {
      uint64_t addr = reinterpret_cast<uint64_t>(p);
      if (addr < imageBase)
        imageBase = addr;
    }
  }

  uint64_t imageBase = ~uint64_t(0);
  std::vector<PRUNTIME_FUNCTION> tables;
};

#endif // _WIN32

std::unique_ptr<llvm::RuntimeDyld::MemoryManager> makeMemoryManager() {
#ifdef _WIN32
  return std::make_unique<Win64SEHMemoryManager>();
#else
  const bool reserveAlloc =
      llvm::Triple(llvm::sys::getProcessTriple()).isAArch64();
  return std::make_unique<llvm::SectionMemoryManager>(nullptr, reserveAlloc);
#endif
}

} // namespace

struct JlcsExecutionEngine {
  std::unique_ptr<llvm::LLVMContext> llvmContext;
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unique_ptr<mlir::SimpleObjectCache> cache;
  std::vector<std::string> functionNames;
  llvm::JITEventListener *gdbListener = nullptr;
  llvm::JITEventListener *perfListener = nullptr;
  llvm::SmallVector<mlir::ExecutionEngine::LibraryDestroyFn> destroyFns;
  bool isInitialized = false;

  ~JlcsExecutionEngine() {
    if (jit)
      llvm::consumeError(jit->deinitialize(jit->getMainJITDylib()));
    for (mlir::ExecutionEngine::LibraryDestroyFn destroy : destroyFns)
      destroy();
  }

  void initialize() {
    if (isInitialized)
      return;
    llvm::cantFail(jit->initialize(jit->getMainJITDylib()));
    isInitialized = true;
  }

  Expected<void *> lookup(StringRef name) const {
    auto expectedSymbol = jit->lookup(name);
    if (!expectedSymbol) {
      std::string errorMessage;
      llvm::raw_string_ostream os(errorMessage);
      llvm::handleAllErrors(expectedSymbol.takeError(),
                            [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
      return makeStringError(errorMessage);
    }
    if (void *fptr = expectedSymbol->toPtr<void *>())
      return fptr;
    return makeStringError("looked up function is null");
  }

  Expected<void (*)(void **)> lookupPacked(StringRef name) const {
    auto result = lookup(packedName(name));
    if (!result)
      return result.takeError();
    return reinterpret_cast<void (*)(void **)>(*result);
  }

  Error invokePacked(StringRef name, void **args) {
    initialize();
    auto expectedFPtr = lookupPacked(name);
    if (!expectedFPtr)
      return expectedFPtr.takeError();
    (*expectedFPtr)(args);
    return Error::success();
  }

  void registerSymbol(StringRef name, void *addr) {
    auto &mainJD = jit->getMainJITDylib();
    llvm::cantFail(mainJD.define(llvm::orc::absoluteSymbols(
        llvm::orc::SymbolMap{{MangleAndInterner(mainJD.getExecutionSession(),
                                                jit->getDataLayout())(name),
                              {llvm::orc::ExecutorAddr::fromPtr(addr),
                               llvm::JITSymbolFlags::Exported}}})));
  }

  void dumpToObjectFile(StringRef filename) {
    if (!cache) {
      llvm::errs() << "cannot dump ExecutionEngine object code to file: "
                      "object cache is disabled\n";
      return;
    }
    if (cache->isEmpty()) {
      for (std::string &functionName : functionNames) {
        auto result = lookupPacked(functionName);
        if (!result) {
          llvm::errs() << "Could not compile " << functionName << ":\n  "
                       << result.takeError() << "\n";
          return;
        }
      }
    }
    cache->dumpToObjectFile(filename);
  }
};

llvm::Expected<JlcsExecutionEngine *>
jlcsEngineCreate(mlir::ModuleOp op, int optLevel, bool dumpObject,
                 llvm::ArrayRef<llvm::StringRef> sharedLibPaths) {
  std::unique_ptr<JlcsExecutionEngine> engine(new JlcsExecutionEngine());
  engine->gdbListener = llvm::JITEventListener::createGDBRegistrationListener();
  if (std::getenv("REPLIBUILD_JIT_PROFILE") != nullptr) {
    if (auto *listener = llvm::JITEventListener::createPerfJITEventListener())
      engine->perfListener = listener;
    else if (auto *listener =
                 llvm::JITEventListener::createIntelJITEventListener())
      engine->perfListener = listener;
  }
  if (dumpObject)
    engine->cache = std::make_unique<mlir::SimpleObjectCache>();

  if (dumpObject) {
    for (auto funcOp : op.getOps<mlir::LLVM::LLVMFuncOp>()) {
      if (funcOp.getBlocks().empty())
        continue;
      engine->functionNames.push_back(funcOp.getSymName().str());
    }
  }

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(op, *ctx);
  if (!llvmModule)
    return makeStringError("could not convert to LLVM IR");

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError)
    return tmBuilderOrError.takeError();
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError)
    return tmOrError.takeError();
  std::unique_ptr<llvm::TargetMachine> tm = std::move(*tmOrError);

  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tm.get());
  packFunctionArguments(llvmModule.get());

  auto dataLayout = llvmModule->getDataLayout();

  llvm::SmallVector<llvm::SmallString<256>, 4> sharedLibAbsPaths;
  llvm::transform(sharedLibPaths, std::back_inserter(sharedLibAbsPaths),
                  [](StringRef libPath) {
                    llvm::SmallString<256> absPath(libPath.begin(),
                                                   libPath.end());
                    llvm::cantFail(llvm::errorCodeToError(
                        llvm::sys::fs::make_absolute(absPath)));
                    return absPath;
                  });

  llvm::StringMap<void *> exportSymbols;
  llvm::SmallVector<mlir::ExecutionEngine::LibraryDestroyFn> destroyFns;
  llvm::SmallVector<StringRef> jitDyLibPaths;

  for (auto &libPath : sharedLibAbsPaths) {
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        libPath.str().str().c_str());
    void *initSym = lib.getAddressOfSymbol(
        mlir::ExecutionEngine::kLibraryInitFnName);
    void *destroySym = lib.getAddressOfSymbol(
        mlir::ExecutionEngine::kLibraryDestroyFnName);
    if (!initSym || !destroySym) {
      jitDyLibPaths.push_back(libPath);
      continue;
    }
    auto initFn =
        reinterpret_cast<mlir::ExecutionEngine::LibraryInitFn>(initSym);
    initFn(exportSymbols);
    destroyFns.push_back(
        reinterpret_cast<mlir::ExecutionEngine::LibraryDestroyFn>(destroySym));
  }
  engine->destroyFns = std::move(destroyFns);

  const llvm::Triple targetTriple = llvmModule->getTargetTriple();

  auto objectLinkingLayerCreator =
      [&](ExecutionSession &session)
      -> Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
    auto objectLayer = std::make_unique<RTDyldObjectLinkingLayer>(
        session, [](const llvm::MemoryBuffer &) { return makeMemoryManager(); });

    if (engine->gdbListener)
      objectLayer->registerJITEventListener(*engine->gdbListener);
    if (engine->perfListener)
      objectLayer->registerJITEventListener(*engine->perfListener);

    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
      // Make sure .pdata/.xdata are actually allocated. They are not
      // "required for execution" in RuntimeDyld's sense — nothing relocates
      // *to* them — so the default filter can drop the only sections the
      // memory manager knows how to register.
      objectLayer->setProcessAllSections(true);
    }

    for (auto &libPath : jitDyLibPaths) {
      auto mb = llvm::MemoryBuffer::getFile(libPath);
      if (!mb) {
        llvm::errs() << "Failed to create MemoryBuffer for: " << libPath
                     << "\nError: " << mb.getError().message() << "\n";
        continue;
      }
      auto &jd = session.createBareJITDylib(std::string(libPath));
      auto loaded = llvm::orc::DynamicLibrarySearchGenerator::Load(
          libPath.str().c_str(), dataLayout.getGlobalPrefix());
      if (!loaded) {
        llvm::errs() << "Could not load " << libPath << ":\n  "
                     << loaded.takeError() << "\n";
        continue;
      }
      jd.addGenerator(std::move(*loaded));
      llvm::cantFail(objectLayer->add(jd, std::move(mb.get())));
    }

    return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(objectLayer));
  };

  auto compileFunctionCreator =
      [&](llvm::orc::JITTargetMachineBuilder jtmb)
      -> Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    jtmb.setCodeGenOptLevel(static_cast<llvm::CodeGenOptLevel>(optLevel));
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
        std::move(tm), engine->cache.get());
  };

  auto jitOrError = llvm::orc::LLJITBuilder()
                        .setCompileFunctionCreator(compileFunctionCreator)
                        .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                        .setDataLayout(dataLayout)
                        .create();
  if (!jitOrError)
    return jitOrError.takeError();

  ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (Error err = (*jitOrError)->addIRModule(std::move(tsm)))
    return err;
  engine->jit = std::move(*jitOrError);

  llvm::orc::JITDylib &mainJD = engine->jit->getMainJITDylib();
  mainJD.addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  // Bind the C++ EH runtime by absolute address. GetForCurrentProcess uses
  // GetProcAddress, which only sees DLL exports — and on mingw-SEH the
  // personality is `__gxx_personality_seh0` in libc++.dll, not v0. Defining
  // them here means try_call IR does not depend on the caller having already
  // dlopened the right C++ runtime and registered the names.
#ifdef _WIN32
  llvm::sys::DynamicLibrary::LoadLibraryPermanently("libc++.dll");
#endif
  {
    MangleAndInterner mangle(mainJD.getExecutionSession(), dataLayout);
    auto defineAbs = [&](StringRef irName, const char *search) {
      void *p = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(search);
      if (!p)
        return;
      llvm::consumeError(mainJD.define(llvm::orc::absoluteSymbols(
          {{mangle(irName),
            {llvm::orc::ExecutorAddr::fromPtr(p),
             llvm::JITSymbolFlags::Exported}}})));
    };
#ifdef _WIN32
    defineAbs("__gxx_personality_seh0", "__gxx_personality_seh0");
    defineAbs("__gxx_personality_v0", "__gxx_personality_seh0");
#else
    defineAbs("__gxx_personality_v0", "__gxx_personality_v0");
#endif
    defineAbs("__cxa_begin_catch", "__cxa_begin_catch");
    defineAbs("__cxa_end_catch", "__cxa_end_catch");
    defineAbs("jlcs_catch_current_exception", "jlcs_catch_current_exception");
    defineAbs("jlcs_set_pending_exception", "jlcs_set_pending_exception");
  }

  auto runtimeSymbolMap = [&](MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    for (auto &exportSymbol : exportSymbols)
      symbolMap[interner(exportSymbol.getKey())] = {
          llvm::orc::ExecutorAddr::fromPtr(exportSymbol.getValue()),
          llvm::JITSymbolFlags::Exported};
    return symbolMap;
  };
  llvm::cantFail(mainJD.define(
      llvm::orc::absoluteSymbols(runtimeSymbolMap(MangleAndInterner(
          mainJD.getExecutionSession(), engine->jit->getDataLayout())))));

  return engine.release();
}

void jlcsEngineDestroy(JlcsExecutionEngine *engine) { delete engine; }

llvm::Expected<void *> jlcsEngineLookup(JlcsExecutionEngine *engine,
                                        llvm::StringRef name) {
  return engine->lookup(name);
}

llvm::Error jlcsEngineInvokePacked(JlcsExecutionEngine *engine,
                                   llvm::StringRef name, void **args) {
  return engine->invokePacked(name, args);
}

void jlcsEngineRegisterSymbol(JlcsExecutionEngine *engine, llvm::StringRef name,
                              void *addr) {
  engine->registerSymbol(name, addr);
}

void jlcsEngineDumpToObjectFile(JlcsExecutionEngine *engine,
                                llvm::StringRef filename) {
  engine->dumpToObjectFile(filename);
}
