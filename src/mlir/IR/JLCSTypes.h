//===- JLCSTypes.h - JLCS dialect types -------------------------*- C++ -*-===//
//
// Type system for the JLCS dialect
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_JLCS_JLCSTYPES_H
#define MLIR_DIALECT_JLCS_JLCSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

// Forward declare types
namespace mlir {
namespace jlcs {
class CStructType;
class ArrayViewType;
} // namespace jlcs
} // namespace mlir

// Manual storage class definitions (LLVM 21 compatible)
namespace mlir {
namespace jlcs {
namespace detail {

struct CStructTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::ArrayRef<Type>, ArrayAttr, bool>;

  CStructTypeStorage(StringAttr juliaTypeName, ::llvm::ArrayRef<Type> fieldTypes, ArrayAttr fieldOffsets, bool isPacked)
      : juliaTypeName(juliaTypeName), fieldTypes(fieldTypes), fieldOffsets(fieldOffsets), isPacked(isPacked) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(juliaTypeName, fieldTypes, fieldOffsets, isPacked);
  }

  static ::llvm::hash_code hashKey(const KeyTy &key) {
    return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));
  }

  static CStructTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    auto juliaTypeName = std::get<0>(key);
    auto fieldTypes = allocator.copyInto(std::get<1>(key));
    auto fieldOffsets = std::get<2>(key);
    auto isPacked = std::get<3>(key);
    return new (allocator.allocate<CStructTypeStorage>())
        CStructTypeStorage(juliaTypeName, fieldTypes, fieldOffsets, isPacked);
  }

  StringAttr juliaTypeName;
  ::llvm::ArrayRef<Type> fieldTypes;
  ArrayAttr fieldOffsets;
  bool isPacked;
};

struct ArrayViewTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::pair<Type, unsigned>;

  ArrayViewTypeStorage(Type elementType, unsigned rank)
      : elementType(elementType), rank(rank) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, rank);
  }

  static ::llvm::hash_code hashKey(const KeyTy &key) {
    return ::llvm::hash_combine(key.first, key.second);
  }

  static ArrayViewTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<ArrayViewTypeStorage>())
        ArrayViewTypeStorage(key.first, key.second);
  }

  Type elementType;
  unsigned rank;
};

} // namespace detail
} // namespace jlcs
} // namespace mlir

// Include generated type class declarations
#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.h.inc"

#endif // MLIR_DIALECT_JLCS_JLCSTYPES_H
