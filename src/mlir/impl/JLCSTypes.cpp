//===- JLCSTypes.cpp - JLCS dialect type implementations ------------------===//
//
// Type implementation methods - Storage classes manually defined in header
//
//===----------------------------------------------------------------------===//

#include "JLCSTypes.h"
#include "JLCSDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::jlcs;

// Include generated type method implementations
// Note: Storage classes are already defined in JLCSTypes.h,
// so redefinitions in .cpp.inc will be ignored by the linker
#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.cpp.inc"

// Manual accessor implementations for CStructType (required due to manual storage)
StringAttr CStructType::getJuliaTypeName() const {
  return getImpl()->juliaTypeName;
}

::llvm::ArrayRef<Type> CStructType::getFieldTypes() const {
  return getImpl()->fieldTypes;
}

ArrayAttr CStructType::getFieldOffsets() const {
  return getImpl()->fieldOffsets;
}

bool CStructType::getIsPacked() const {
  return getImpl()->isPacked;
}

// Same reason for ArrayViewType — `genStorageClass = 0` means TableGen emits
// the DECLARATIONS but not the bodies, because the bodies it would write reach
// into a storage class it did not generate. CStructType's were written; these
// were not, so `libJLCS.so` shipped with them undefined. Nothing called them
// (`jlcs.array_view` is only reached through the lowering, which reads the
// storage directly), so the gap survived every test: RTLD_LAZY binds on first
// call, and there was no first call.
Type ArrayViewType::getElementType() const {
  return getImpl()->elementType;
}

unsigned ArrayViewType::getRank() const {
  return getImpl()->rank;
}