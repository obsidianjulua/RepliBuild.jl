//===- JLCSLoweringUtils.h - Utilities for JLCS lowering --------*- C++ -*-===//
//
// Utility functions for lowering JLCS dialect ops to LLVM dialect
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_JLCS_JLCSLOWERINGUTILS_H
#define MLIR_DIALECT_JLCS_JLCSLOWERINGUTILS_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace jlcs {

/// Helper function to load a field from a struct by byte offset.
/// Used for accessing fields in C-ABI structs and ArrayView descriptors.
///
/// Performs: *(T*)((i8*)structPtr + byteOffset)
///
/// @param loc Location for generated ops
/// @param rewriter Pattern rewriter for creating new ops
/// @param structPtr Pointer to the struct
/// @param byteOffset Byte offset of the field within the struct
/// @param targetType Expected type of the loaded value
/// @return The loaded value
inline Value getStructField(Location loc, ConversionPatternRewriter &rewriter,
                             Value structPtr, int64_t byteOffset,
                             Type targetType) {
  auto ctx = rewriter.getContext();

  // 1. Create byte offset constant
  Value offset = rewriter.create<arith::ConstantIntOp>(loc, byteOffset, 64);

  // 2. GEP to field address (using i8* as universal pointer)
  Type i8PtrType = LLVM::LLVMPointerType::get(ctx);
  Value fieldAddrI8 = rewriter.create<LLVM::GEPOp>(
      loc, i8PtrType, rewriter.getI8Type(), structPtr,
      ArrayRef<LLVM::GEPArg>({offset}));

  // 3. Load the value directly (opaque pointers, no bitcast needed)
  Value loadedVal = rewriter.create<LLVM::LoadOp>(loc, targetType, fieldAddrI8);

  return loadedVal;
}

/// Helper function to store a value to a struct field by byte offset.
///
/// Performs: *(T*)((i8*)structPtr + byteOffset) = value
///
/// @param loc Location for generated ops
/// @param rewriter Pattern rewriter for creating new ops
/// @param structPtr Pointer to the struct
/// @param byteOffset Byte offset of the field within the struct
/// @param value Value to store
inline void setStructField(Location loc, ConversionPatternRewriter &rewriter,
                            Value structPtr, int64_t byteOffset, Value value) {
  auto ctx = rewriter.getContext();

  // 1. Create byte offset constant
  Value offset = rewriter.create<arith::ConstantIntOp>(loc, byteOffset, 64);

  // 2. GEP to field address
  Type i8PtrType = LLVM::LLVMPointerType::get(ctx);
  Value fieldAddrI8 = rewriter.create<LLVM::GEPOp>(
      loc, i8PtrType, rewriter.getI8Type(), structPtr,
      ArrayRef<LLVM::GEPArg>({offset}));

  // 3. Store the value
  rewriter.create<LLVM::StoreOp>(loc, value, fieldAddrI8);
}

} // namespace jlcs
} // namespace mlir

#endif // MLIR_DIALECT_JLCS_JLCSLOWERINGUTILS_H
