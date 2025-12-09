# JLCS MLIR Dialect - Complete Guide for Julia Developers

> **Building MLIR dialects from Julia: A practical guide using the JLCS (Julia C-Struct) dialect**

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Understanding MLIR from a Julia Perspective](#understanding-mlir-from-a-julia-perspective)
5. [Project Structure](#project-structure)
6. [Building the Dialect](#building-the-dialect)
7. [Using from Julia](#using-from-julia)
8. [Deep Dive: How It All Works](#deep-dive-how-it-all-works)
9. [Creating Your Own Dialect](#creating-your-own-dialect)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

This guide teaches Julia developers how to create custom MLIR dialects for advanced FFI scenarios. The JLCS dialect demonstrates:

- **C-ABI struct manipulation** (field access by byte offset)
- **Virtual method dispatch** (vtable-based calls)
- **Strided array operations** (cross-language arrays)
- **Complete LLVM lowering** (executable code generation)
- **JIT compilation** (runtime code execution)

### Why MLIR for Julia FFI?

Traditional Julia FFI (`ccall`) works well for simple C functions, but struggles with:
- C++ virtual methods and inheritance
- Complex struct layouts with padding
- STL containers with implementation-defined layouts
- Cross-language optimization opportunities

MLIR provides:
- **Custom IR** tailored to your FFI needs
- **Transformation passes** for optimization
- **Direct LLVM lowering** for native performance
- **Type-safe operations** verified at IR level

---

## Prerequisites

### System Requirements

- **LLVM/MLIR 18+** (system installation required)
- **CMake 3.20+**
- **C++17 compiler** (GCC/Clang)
- **Julia 1.9+**

### Installing MLIR

#### Arch Linux
```bash
yay -S mlir llvm
```

#### Ubuntu/Debian
```bash
apt install llvm-18-dev mlir-18-dev
```

#### macOS
```bash
brew install llvm
```

### Verify Installation

```bash
# Check mlir-tblgen (TableGen tool)
mlir-tblgen --version

# Check llvm-config
llvm-config --version

# Check MLIR CMake modules
llvm-config --cmakedir
```

---

## Quick Start

### 1. Build the Dialect

```bash
cd examples/Mlir
./build_dialect.sh
```

This will:
1. Check for MLIR installation
2. Run TableGen to generate C++ from `.td` files
3. Compile the dialect into `build/libJLCS.so`

### 2. Test from Julia

```bash
julia
```

```julia
# Load the MLIR bindings. Use RepliBuild.jl to compile ffi bindings to get c++ types and structure for faster workflow.
include("../../src/MLIRNative.jl")
using .MLIRNative

# Run the test suite
MLIRNative.test_dialect()
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

TODO: Create jlcs.type_info operation

Cleaning up... ✓

======================================================================
 All tests passed!
======================================================================
```

---

## Understanding MLIR from a Julia Perspective

### MLIR Concepts → Julia Equivalents

| MLIR Concept | Julia Equivalent | Purpose |
|--------------|------------------|---------|
| **Dialect** | Module/Package | Namespace for operations and types |
| **Operation** | Function/Method | Computation unit (e.g., `add`, `call`) |
| **Type** | Julia Type | Data representation (e.g., `Int64`, `Ptr`) |
| **Attribute** | Constant/Metadata | Compile-time values (offsets, names) |
| **Region** | Code Block | Contains multiple operations |
| **Pass** | Transformation | Rewrites IR (like Julia lowering) |

### MLIR IR vs Julia IR

**Julia IR (Typed AST)**
```julia
# Julia function
function get_field(obj::Ptr{MyStruct}, offset::Int)
    unsafe_load(Ptr{Int64}(obj + offset))
end

# Julia IR (simplified)
%1 = add(%obj, %offset)     # pointer arithmetic
%2 = load(%1)               # memory load
return %2
```

**MLIR IR (JLCS Dialect)**
```mlir
// High-level operation
%result = jlcs.get_field %obj_ptr, 16 : i64

// After lowering to LLVM dialect
%offset = arith.constant 16 : i64
%field_addr = llvm.getelementptr %obj_ptr[%offset] : (!llvm.ptr, i64) -> !llvm.ptr
%result = llvm.load %field_addr : !llvm.ptr -> i64
```

**Key Difference**: MLIR lets you define custom operations (`jlcs.get_field`) at a higher abstraction level, then lower them progressively to LLVM.

---

## Project Structure

```
examples/Mlir/
├── JLCS.td                      # TableGen dialect definition (START HERE)
├── CMakeLists.txt               # Build configuration
├── build_dialect.sh             # Build script
│
├── IR/                          # C++ Headers
│   ├── JLCSDialect.h           # Dialect declaration
│   ├── JLCSOps.h               # Operations header
│   ├── JLCSTypes.h             # Types header
│   └── JLCSLoweringUtils.h     # Helper functions for lowering
│
├── src/                         # C++ Implementation
│   ├── JLCSDialect.cpp         # Dialect registration
│   ├── JLCSOps.cpp             # Operation implementations
│   ├── JLCSTypes.cpp           # Type storage implementations
│   ├── JLCSPasses.cpp          # LLVM lowering pass
│   └── JLCSCHelpers.cpp        # C API for Julia
│
└── build/                       # Generated files (after build)
    ├── libJLCS.so              # Compiled dialect library
    ├── JLCSDialect.h.inc       # Generated from TableGen
    ├── JLCSOps.h.inc           # Generated operation declarations
    └── ...
```

### File Roles

| File | Language | Purpose |
|------|----------|---------|
| `JLCS.td` | TableGen | **Source of truth** - defines dialect schema |
| `*.cpp` | C++ | Implements behavior (lowering, verification) |
| `*.h` | C++ | Headers that include TableGen-generated code |
| `JLCSCHelpers.cpp` | C | Exports C API for Julia `ccall` |

---

## Building the Dialect

### Build Process Overview

```
JLCS.td  ──→  mlir-tblgen  ──→  *.h.inc / *.cpp.inc
                                       ↓
                                 C++ Compiler
                                       ↓
                                  libJLCS.so  ──→  Julia ccall
```

### Step-by-Step Build

#### 1. TableGen Generation

CMake automatically runs `mlir-tblgen` with different flags:

```cmake
mlir_tablegen(JLCSDialect.h.inc -gen-dialect-decls)
mlir_tablegen(JLCSDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(JLCSOps.h.inc -gen-op-decls)
mlir_tablegen(JLCSOps.cpp.inc -gen-op-defs)
mlir_tablegen(JLCSTypes.h.inc -gen-type-decls)
mlir_tablegen(JLCSTypes.cpp.inc -gen-type-defs)
```

This reads `JLCS.td` and generates C++ code.

#### 2. Compilation

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Produces: `libJLCS.so` (shared library)

#### 3. Manual Build (if script fails)

```bash
mkdir -p build && cd build

# Find LLVM/MLIR paths
LLVM_DIR=$(llvm-config --cmakedir)
MLIR_DIR=$(llvm-config --prefix)/lib/cmake/mlir

# Configure
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR"

# Build
make -j$(nproc)
```

### Verifying the Build

```bash
# Check library exists
ls -lh build/libJLCS.so

# Check symbols
nm -D build/libJLCS.so | grep registerJLCSDialect
```

Should see: `registerJLCSDialect` exported.

---

## Using from Julia

### Architecture: Julia ↔ MLIR

```
Julia Code
    ↓ (ccall)
MLIR C API (libMLIR.so)
    ↓
JLCS Dialect (libJLCS.so)
    ↓
LLVM IR
    ↓
Native Code
```

### Basic Julia Bindings

The [`MLIRNative.jl`](../../src/MLIRNative.jl) module provides the interface:

```julia
module MLIRNative

# Opaque types for MLIR objects
const MlirContext = Ptr{Cvoid}
const MlirModule = Ptr{Cvoid}
const MlirOperation = Ptr{Cvoid}

# Library paths
const libMLIR = "libMLIR"  # System MLIR C API
const libJLCS_path = joinpath(@__DIR__, "Mlir", "build", "libJLCS.so")

# Create MLIR context
function create_context()
    # Call MLIR C API
    ctx = ccall((:mlirContextCreate, libMLIR), MlirContext, ())

    # Load JLCS dialect dynamically
    dlopen(libJLCS_path, RTLD_GLOBAL)

    # Register JLCS dialect
    ccall((:registerJLCSDialect, libJLCS_path), Cvoid, (MlirContext,), ctx)

    return ctx
end

end
```

### Step-by-Step Usage

#### 1. Create MLIR Context

```julia
ctx = MLIRNative.create_context()
```

A **context** holds all MLIR state (types, operations, dialects).

#### 2. Create Empty Module

```julia
mod = MLIRNative.create_module(ctx)
```

A **module** is the top-level container for MLIR operations.

#### 3. Print Module IR

```julia
MLIRNative.print_module(mod)
```

Output:
```mlir
module {
}
```

#### 4. Cleanup

```julia
MLIRNative.destroy_context(ctx)
```

### Using the Convenience Macro

```julia
MLIRNative.@with_context begin
    mod = create_module(ctx)
    print_module(mod)
    # ctx automatically cleaned up
end
```

---

## Deep Dive: How It All Works

### Part 1: TableGen Definition

**File**: `JLCS.td`

TableGen is a domain-specific language for defining MLIR dialects. Think of it as "code generation configuration."

#### Defining the Dialect

```tablegen
def JLCS_Dialect : Dialect {
  let name = "jlcs";
  let cppNamespace = "::mlir::jlcs";
  let summary = "Julia C-Struct layout & FFE dialect";
  let description = [{
    Dialect for modeling C-layouted Julia structs and FFE semantics.
  }];
}
```

**Generated C++**:
```cpp
namespace mlir {
namespace jlcs {

class JLCSDialect : public ::mlir::Dialect {
public:
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("jlcs");
  }
  // ... more generated code
};

} // namespace jlcs
} // namespace mlir
```

#### Defining a Type

```tablegen
def CStructType : JLCS_Type<"CStruct", "c_struct"> {
  let summary = "C-ABI-compatible struct";
  let parameters = (ins
    "StringAttr":$juliaTypeName,
    ArrayRefParameter<"Type", "field types">:$fieldTypes,
    "ArrayAttr":$fieldOffsets
  );
  let assemblyFormat = "`<` $juliaTypeName `,` `[` $fieldTypes `]` `,` `[` $fieldOffsets `]` `>`";
}
```

**Usage in MLIR IR**:
```mlir
!jlcs.c_struct<"MyStruct", [i64, f64], [0, 8]>
```

**Julia equivalent concept**:
```julia
struct CStructType
    julia_type_name::String
    field_types::Vector{Type}
    field_offsets::Vector{Int}
end
```

#### Defining an Operation

```tablegen
def GetFieldOp : JLCS_Op<"get_field"> {
  let summary = "Read a field from a C-compatible struct";
  let arguments = (ins
    AnyType:$structValue,
    I64Attr:$fieldOffset
  );
  let results = (outs AnyType:$result);
}
```

**Usage in MLIR IR**:
```mlir
%result = jlcs.get_field %struct_ptr, 16 : i64
```

**Julia concept**:
```julia
function get_field(struct_value::Ptr, field_offset::Int64)::Int64
    # Implementation defined in lowering pass
end
```

### Part 2: C++ Implementation

#### Dialect Registration

**File**: `src/JLCSDialect.cpp`

```cpp
void JLCSDialect::initialize() {
    // Register all operations defined in TableGen
    addOperations<
#define GET_OP_LIST
#include "JLCSOps.cpp.inc"
        >();

    // Register all types defined in TableGen
    addTypes<
#define GET_TYPEDEF_LIST
#include "JLCSTypes.cpp.inc"
        >();
}
```

This includes the generated code from `mlir-tblgen`.

#### Lowering Pass

**File**: `src/JLCSPasses.cpp`

Lowering transforms high-level JLCS operations into low-level LLVM operations:

```cpp
struct GetFieldOpLowering : public ConversionPattern {
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto getFieldOp = cast<GetFieldOp>(op);

        // Get inputs
        Value structPtr = operands[0];
        int64_t byteOffset = getFieldOp.getFieldOffset();

        // Generate LLVM IR:
        // 1. Create constant for byte offset
        Value offset = rewriter.create<arith::ConstantIntOp>(
            loc, byteOffset, 64);

        // 2. GEP (pointer arithmetic)
        Value fieldAddr = rewriter.create<LLVM::GEPOp>(
            loc, ptrType, i8Type, structPtr, offset);

        // 3. Load value
        Value result = rewriter.create<LLVM::LoadOp>(
            loc, resultType, fieldAddr);

        // Replace original operation
        rewriter.replaceOp(op, result);
        return success();
    }
};
```

**Before lowering** (JLCS dialect):
```mlir
%result = jlcs.get_field %obj_ptr, 16 : i64
```

**After lowering** (LLVM dialect):
```mlir
%offset = arith.constant 16 : i64
%field_addr = llvm.getelementptr %obj_ptr[%offset] : (!llvm.ptr, i64) -> !llvm.ptr
%result = llvm.load %field_addr : !llvm.ptr -> i64
```

### Part 3: Julia Bindings

#### C API Wrapper

**File**: `src/JLCSCHelpers.cpp`

Export C functions that Julia can call:

```cpp
extern "C" {

// Wrap MLIR C++ API in C functions
MlirContext jlcs_create_context() {
    auto *ctx = new MLIRContext();
    return wrap(ctx);  // Convert C++ → C pointer
}

void jlcs_destroy_context(MlirContext context) {
    delete unwrap(context);  // Convert C → C++ pointer
}

}
```

#### Julia ccall

**File**: `src/MLIRNative.jl`

```julia
function create_context()
    # Call C function via ccall
    ctx = ccall(
        (:mlirContextCreate, libMLIR),  # (symbol, library)
        MlirContext,                     # return type
        ()                               # argument types
    )

    # Register JLCS dialect
    ccall(
        (:registerJLCSDialect, libJLCS_path),
        Cvoid,           # return type
        (MlirContext,),  # argument types
        ctx              # arguments
    )

    return ctx
end
```

---

## Creating Your Own Dialect

### Step 1: Fork the JLCS Dialect

```bash
cp -r examples/Mlir examples/MyDialect
cd examples/MyDialect
```

### Step 2: Rename Files

```bash
# Rename all JLCS → MyDialect
find . -name "*JLCS*" -exec rename 's/JLCS/MyDialect/' {} \;

# Update file contents
sed -i 's/JLCS/MyDialect/g' *.td
sed -i 's/jlcs/mydialect/g' *.td
sed -i 's/JLCS/MyDialect/g' src/*.cpp IR/*.h
sed -i 's/jlcs/mydialect/g' src/*.cpp IR/*.h
```

### Step 3: Define Your Operations

Edit `MyDialect.td`:

```tablegen
// Define your custom operation
def MyCustomOp : MyDialect_Op<"my_op"> {
  let summary = "Description of what this operation does";

  let arguments = (ins
    I64:$input1,
    F64:$input2
  );

  let results = (outs
    I64:$output
  );

  // Optional: custom assembly format
  let assemblyFormat = "$input1 `,` $input2 attr-dict `:` type($output)";
}
```

### Step 4: Implement Lowering

Edit `src/MyDialectPasses.cpp`:

```cpp
struct MyCustomOpLowering : public ConversionPattern {
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto myOp = cast<MyCustomOp>(op);
        Location loc = myOp.getLoc();

        // Your lowering logic here
        // Convert mydialect.my_op → LLVM operations

        rewriter.replaceOp(op, result);
        return success();
    }
};
```

### Step 5: Build and Test

```bash
./build_dialect.sh

# Test from Julia
julia -e 'include("../../src/MLIRNative.jl"); using .MLIRNative; test_dialect()'
```

---

## Troubleshooting

### Build Issues

#### Error: `mlir-tblgen not found`

**Solution**: Install MLIR system package
```bash
# Arch
yay -S mlir

# Ubuntu
apt install mlir-18-dev
```

#### Error: `Could not find MLIR CMake modules`

**Solution**: Set MLIR_DIR manually
```bash
export MLIR_DIR=/usr/lib/cmake/mlir
cmake .. -DMLIR_DIR=$MLIR_DIR
```

#### Error: `undefined reference to mlir::Dialect::~Dialect()`

**Solution**: Link order issue - ensure MLIR libraries are linked
```cmake
target_link_libraries(MyDialect PUBLIC MLIRIR MLIRSupport)
```

### Runtime Issues

#### Error: `JLCS dialect library not found`

**Solution**: Check library path
```julia
isfile(MLIRNative.libJLCS_path)  # Should return true
```

#### Error: `cannot open shared object file`

**Solution**: Set LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
```

#### Error: `symbol lookup error: undefined symbol`

**Solution**: Register dialect before use
```julia
dlopen(libJLCS_path, RTLD_GLOBAL)  # Load symbols globally
ccall((:registerJLCSDialect, libJLCS_path), Cvoid, (MlirContext,), ctx)
```

---

## Next Steps

1. **Read the TableGen guide**: Learn the `.td` language in depth
2. **Study the lowering passes**: Understand MLIR transformations
3. **Explore MLIR dialects**: See how `arith`, `llvm`, `func` work
4. **Build your own**: Create a dialect for your FFI use case

### Additional Resources

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [TableGen Programmer's Reference](https://llvm.org/docs/TableGen/)
- [Writing MLIR Passes](https://mlir.llvm.org/docs/PassManagement/)
- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/)

### Example Use Cases

- **Database FFI**: Custom operations for SQL queries
- **GPU Kernels**: High-level operations that lower to CUDA/ROCm
- **DSP**: Signal processing operations with SIMD lowering
- **Linear Algebra**: BLAS-like operations with auto-tuning

---

**Questions?** Open an issue on the RepliBuild.jl repository!
