//===- JLCSOps.h - JLCS dialect operations ----------------------*- C++ -*-===//
//
// Operations for the JLCS dialect
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_JLCS_JLCSOPS_H
#define MLIR_DIALECT_JLCS_JLCSOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "JLCSOps.h.inc"

#endif // MLIR_DIALECT_JLCS_JLCSOPS_H
