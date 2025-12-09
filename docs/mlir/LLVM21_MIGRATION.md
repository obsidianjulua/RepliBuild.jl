# LLVM 21 Migration Guide for JLCS Dialect

This document records the solutions to all compatibility issues encountered when building the JLCS MLIR dialect with LLVM 21.

## Overview

The JLCS dialect successfully builds and runs with **LLVM 21.1.6** (Arch Linux). Several API changes from earlier LLVM versions required specific solutions.

## Issues and Solutions

### 1. TableGen Flag Renaming

**Issue**: TableGen command-line flags were renamed in LLVM 21.

**Error**:
```
mlir-tblgen: Unknown command line argument '-gen-type-decls'
```

**Solution**: Update `CMakeLists.txt` with new flag names:
```cmake
mlir_tablegen(JLCSTypes.h.inc -gen-typedef-decls)     # was: -gen-type-decls
mlir_tablegen(JLCSTypes.cpp.inc -gen-typedef-defs)    # was: -gen-type-defs
mlir_tablegen(JLCSInterfaces.h.inc -gen-type-interface-decls)  # was: -gen-interface-decls
mlir_tablegen(JLCSInterfaces.cpp.inc -gen-type-interface-defs) # was: -gen-interface-defs
```

**Files Modified**: [src/mlir/CMakeLists.txt:40-47](../../src/mlir/CMakeLists.txt#L40-L47)

---

### 2. Type Storage Visibility (CRITICAL)

**Issue**: LLVM 21's type registration requires complete storage class definitions visible when `Dialect::addTypes<>()` is called. The `std::is_trivially_destructible_v<Storage>` trait needs the full class, not just a forward declaration.

**Error**:
```
invalid 'static_cast' from type 'mlir::StorageUniquer::BaseStorage*'
to type 'mlir::jlcs::detail::ArrayViewTypeStorage*'
```

**Root Cause**: When `JLCSDialect::initialize()` calls `addTypes<CStructType, ArrayViewType>()`, the compiler checks if the storage is trivially destructible, which requires seeing the complete storage class definition.

**Solution**: Include `GET_TYPEDEF_CLASSES` in the dialect implementation file **before** type registration:

```cpp
// JLCSDialect.cpp
#include "JLCSDialect.h"
#include "JLCSOps.h"
#include "JLCSTypes.h"

// IMPORTANT: Include type storage definitions BEFORE registering
// This makes the complete storage class visible for registration
#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.cpp.inc"

void JLCSDialect::initialize() {
    // Register types (storage already included above)
    addTypes<CStructType, ArrayViewType>();
}
```

**Key Insight**: The storage definition must be in the same translation unit that calls `addTypes<>()`.

**Files Modified**:
- [src/mlir/impl/JLCSDialect.cpp:14-28](../../src/mlir/impl/JLCSDialect.cpp#L14-L28) - Added storage inclusion
- [src/mlir/impl/JLCSTypes.cpp](../../src/mlir/impl/JLCSTypes.cpp) - Now empty (storage moved)

---

### 3. CallOp API Changes

**Issue**: `LLVM::CallOp::build()` no longer has a simple builder for indirect calls in LLVM 21.

**Error**:
```
no matching function for call to 'mlir::LLVM::CallOp::build'
(24 candidate functions not viable)
```

**Solution**: Use manual `OperationState` construction for indirect calls:

```cpp
// Determine result types
SmallVector<Type, 1> resultTypeVec;
if (vcallOp.getResult()) {
    resultTypeVec.push_back(vcallOp.getResult().getType());
}

// Build CallOp manually with OperationState
SmallVector<Value> allOperands;
allOperands.push_back(funcPtr);  // Function pointer first
allOperands.append(callArgs.begin(), callArgs.end());

OperationState state(loc, LLVM::CallOp::getOperationName());
state.addOperands(allOperands);
state.addTypes(resultTypeVec);
state.addAttribute("callee", FlatSymbolRefAttr());  // empty for indirect

Operation *callOp = rewriter.create(state);
```

**Files Modified**: [src/mlir/impl/JLCSPasses.cpp:137-157](../../src/mlir/impl/JLCSPasses.cpp#L137-L157)

---

### 4. Missing MLIR C API Library

**Issue**: System MLIR installation doesn't provide `libMLIR.so` with C API symbols, only static libraries.

**Error**:
```
could not load symbol "mlirContextCreate": undefined symbol
```

**Solution**: Implement C API wrappers using C++ API in our dialect library:

```cpp
// JLCSCAPIWrappers.cpp
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/CAPI/IR.h"

extern "C" {

MlirContext mlirContextCreate() {
    auto *ctx = new mlir::MLIRContext();
    return wrap(ctx);
}

void mlirContextDestroy(MlirContext context) {
    delete unwrap(context);
}

MlirLocation mlirLocationUnknownGet(MlirContext context) {
    mlir::MLIRContext *ctx = unwrap(context);
    mlir::Location loc = mlir::UnknownLoc::get(ctx);
    return wrap(loc);
}

// ... etc
}
```

Link against static `MLIRCAPIIR` library:
```cmake
LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR  # Static C API library
```

**Files Created**: [src/mlir/impl/JLCSCAPIWrappers.cpp](../../src/mlir/impl/JLCSCAPIWrappers.cpp)
**Files Modified**: [src/mlir/CMakeLists.txt:62,69](../../src/mlir/CMakeLists.txt#L62)

---

### 5. MLIRToLLVMIRTranslation Library Renamed

**Issue**: Library name changed in LLVM 21.

**Error**:
```
/usr/bin/ld: cannot find -lMLIRToLLVMIRTranslation
```

**Solution**: Use the new library name:
```cmake
LINK_LIBS PUBLIC
  MLIRToLLVMIRTranslationRegistration  # was: MLIRToLLVMIRTranslation
```

**Files Modified**: [src/mlir/CMakeLists.txt:82](../../src/mlir/CMakeLists.txt#L82)

---

## Build Instructions

After applying all fixes:

```bash
cd src/mlir
./build.sh
```

Expected output:
```
==============================================
 Build Complete!
==============================================
Library: /path/to/build/libJLCS.so
Size: 19M
```

## Testing

```bash
cd /path/to/RepliBuild.jl
julia --project=. -e 'include("src/MLIRNative.jl"); using .MLIRNative; test_dialect()'
```

Expected output:
```
======================================================================
 JLCS MLIR Dialect Test
======================================================================
Checking library... ✓
Creating MLIR context... ✓
Creating MLIR module... ✓

Empty module:
module {
}

Cleaning up... ✓

======================================================================
 All tests passed!
======================================================================
```

## System Requirements

- **LLVM**: 21.1.6 or later
- **MLIR**: Same version as LLVM
- **CMake**: 3.20+
- **C++ Standard**: C++17

## Key Takeaways

1. **Type Storage Pattern**: Always include `GET_TYPEDEF_CLASSES` in the dialect implementation file before calling `addTypes<>()` in LLVM 21.

2. **C API Integration**: On systems where MLIR C API is only available as static libraries, embed wrappers in your dialect library.

3. **TableGen Flags**: Always check TableGen flag names against the version's documentation.

4. **Manual Operation Construction**: For complex operations like indirect calls, use `OperationState` instead of convenience builders.

## Future Considerations

- Monitor LLVM release notes for API changes
- Consider using MLIR's C API more extensively to isolate from C++ API changes
- Maintain version-specific build configurations if supporting multiple LLVM versions

---

**Status**: ✅ All LLVM 21 compatibility issues resolved
**Date**: 2025-12-09
**Tested On**: LLVM 21.1.6 (Arch Linux)
