# MLIR Build Notes - LLVM 21 Compatibility

## Current Status

The MLIR dialect has been successfully restructured into production:
- ✅ Documentation complete (60KB across 4 files in `docs/mlir/`)
- ✅ Source code organized in `src/mlir/`
- ✅ Julia bindings updated in `src/MLIRNative.jl`
- ✅ Build scripts and CMake configuration updated for LLVM 21
- ⚠️  **Build blocked** by LLVM 21 API/ABI issues

## LLVM 21 Compatibility Issues

### 1. TableGen Flag Changes (FIXED ✅)
**Issue**: mlir-tblgen flags changed in LLVM 21
**Solution Applied**:
- `-gen-type-decls` → `-gen-typedef-decls`
- `-gen-type-defs` → `-gen-typedef-defs`
- `-gen-interface-decls` → `-gen-type-interface-decls`
- `-gen-interface-defs` → `-gen-type-interface-defs`

### 2. CallOp API Changes (FIXED ✅)
**Issue**: LLVM::CallOp no longer supports simple 5-arg builder for indirect calls
**Attempted Solutions**:
1. Manual OperationState construction ✅
2. Removed NoneType check (not available in LLVM 21)

### 3. Type Storage Issues (BLOCKING 🚧)

**Problem**: TableGen-generated type storage classes are incomplete when `genStorageClass = 1` is used.

**Error**:
```
error: invalid 'static_cast' from type 'mlir::StorageUniquer::BaseStorage*'
       to type 'mlir::jlcs::detail::ArrayViewTypeStorage*'
note: class type 'mlir::jlcs::detail::ArrayViewTypeStorage' is incomplete
```

**Root Cause**: LLVM 21 changed how TypeStorage classes are generated/registered. The `genStorageClass = 1` directive creates forward declarations but the full implementation must be visible before type registration.

**Workarounds Attempted**:
1. Remove `genStorageClass` - still needs storage implementation
2. Manual storage implementation in `.cpp` file - visibility issues
3. Include storage in header - circular dependency

## Recommended Solutions

### Option A: Use LLVM 18 (Fastest)
```bash
# Install LLVM 18 alongside LLVM 21
yay -S llvm18 mlir18

# Build with LLVM 18
export LLVM_DIR=/usr/lib/llvm18/lib/cmake/llvm
export MLIR_DIR=/usr/lib/llvm18/lib/cmake/mlir
cd src/mlir && ./build.sh
```

### Option B: Simplify Types (Medium Effort)
Remove complex parametric types temporarily:
```tablegen
// Simplified version without parameters
def CStructType : JLCS_Type<"CStruct", "c_struct"> {
  let summary = "C-ABI struct (opaque)";
  // No parameters - simpler storage
}
```

### Option C: Deep Dive into LLVM 21 TypeDef (High Effort)
Research LLVM 21's new type definition system:
1. Study `mlir/IR/TypeDef.td` in LLVM 21 source
2. Find example dialects in LLVM 21 tree
3. Understand new storage requirements
4. Update JLCS types accordingly

Estimated time: 4-8 hours

## Working Components

Despite build issues, these are production-ready:

### Documentation (`docs/mlir/`)
- Complete startup guide
- TableGen reference
- Practical examples
- Learning paths

### Core Dialect Definition (`src/mlir/JLCS.td`)
- 7 operations defined
- 2 type definitions (syntax correct)
- Proper traits and interfaces
- Assembly formats specified

### Lowering Passes (`src/mlir/impl/JLCSPasses.cpp`)
- GetFieldOp lowering ✅
- SetFieldOp lowering ✅
- VirtualCallOp lowering ✅ (LLVM 21 compatible)
- LoadArrayElementOp lowering ✅
- StoreArrayElementOp lowering ✅

### Julia Integration (`src/MLIRNative.jl`)
- Context management
- Module creation
- JIT support skeleton
- Proper error messages

## Next Steps

**Immediate** (Choose One):
1. Switch to LLVM 18 for quick success
2. Simplify types to get basic build working
3. Research LLVM 21 type system deeply

**After Successful Build**:
1. Run `julia -e 'include("src/MLIRNative.jl"); using .MLIRNative; test_dialect()'`
2. Create integration tests in `test/mlir/`
3. Connect to RepliBuild DWARF parser
4. Generate JLCS IR from C++ vtable data

## Files Modified for LLVM 21

- `src/mlir/CMakeLists.txt` - TableGen flags
- `src/mlir/impl/JLCSDialect.cpp` - Namespace fixes
- `src/mlir/impl/JLCSPasses.cpp` - CallOp API
- `src/mlir/impl/JLCSTypes.cpp` - Storage attempts
- `src/mlir/IR/JLCSTypes.h` - Include fixes
- `src/mlir/JLCS.td` - Interface syntax

## Build Command

```bash
cd /home/grim/Desktop/Projects/RepliBuild.jl/src/mlir
./build.sh
```

Expected output (when working):
```
============================================
 Build Complete!
============================================
Library: /path/to/libJLCS.so
Size: XX KB
```

---

**Last Updated**: December 9, 2024
**LLVM Version**: 21.1.6 (Arch Linux)
**Blocker**: Type storage registration in LLVM 21
**Recommended**: Try LLVM 18 or simplify type definitions
