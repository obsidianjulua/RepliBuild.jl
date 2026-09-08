// JlcsJIT.h — ORC LLJIT owned by the dialect, not mlir::ExecutionEngine.
//
// MLIR's ExecutionEngine::create hardcodes RTDyldObjectLinkingLayer + a
// vanilla SectionMemoryManager. That is the entire object-linking layer, and
// ExecutionEngineOptions has no hook to replace it (sectionMemoryMapper is a
// MemoryMapper*, not a MemoryManager). We build the same LLJIT stack here so
// the memory manager can be swapped on Windows; see JlcsJIT.cpp.

#ifndef JLCS_JIT_H
#define JLCS_JIT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

struct JlcsExecutionEngine;

// Raw pointer: unique_ptr<incomplete> cannot be destroyed from the wrapper TU.
llvm::Expected<JlcsExecutionEngine *>
jlcsEngineCreate(mlir::ModuleOp op, int optLevel, bool dumpObject,
                 llvm::ArrayRef<llvm::StringRef> sharedLibPaths);

void jlcsEngineDestroy(JlcsExecutionEngine *engine);

llvm::Expected<void *> jlcsEngineLookup(JlcsExecutionEngine *engine,
                                        llvm::StringRef name);

llvm::Error jlcsEngineInvokePacked(JlcsExecutionEngine *engine,
                                   llvm::StringRef name, void **args);

void jlcsEngineRegisterSymbol(JlcsExecutionEngine *engine, llvm::StringRef name,
                              void *addr);

void jlcsEngineDumpToObjectFile(JlcsExecutionEngine *engine,
                                llvm::StringRef filename);

#endif // JLCS_JIT_H
