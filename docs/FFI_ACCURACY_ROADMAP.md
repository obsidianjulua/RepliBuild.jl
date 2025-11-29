# C/C++ FFI Accuracy Roadmap

**FOCUS**: Perfect C/C++ → Julia type mapping and binding generation

**STATUS**: Core functionality works (27/27 tests pass), but accuracy issues remain

---

## Current State Assessment

### ✅ What Works
- **Basic compilation**: C++ → LLVM IR → .so library
- **Symbol extraction**: Functions discovered via `nm`
- **Basic wrappers**: Generic `ccall()` wrappers generated
- **DWARF extraction**: Debug info parsed for types
- **Struct extraction**: Basic struct layout from DWARF
- **Test suite**: 27 tests passing

### ❌ What Needs Fixing
Based on analysis of `test/test_advanced_types.cpp` → generated bindings:

1. **Enum Extraction & Mapping** - Enums not extracted or mapped to `@enum`
2. **Array Dimensions** - Multi-dimensional arrays not handled correctly
3. **Function Pointers** - Not detected or typed properly
4. **Parameter Types** - Generic `Any` instead of specific types
5. **Return Types** - Some return `Any` instead of proper types
6. **Struct Member Types** - Members showing as `Any` instead of concrete types

---

## Test Case: test_advanced_types.cpp

This file comprehensively tests the edge cases we need to handle:

### Enums (3 types)
```cpp
enum Color { Red = 0, Green = 1, Blue = 2 };
enum class Status : unsigned int { Idle = 0, Running = 100, ... };
enum class Direction : int { North = 1, South = -1, ... };
```

**Expected Julia:**
```julia
@enum Color::Cuint begin
    Red = 0
    Green = 1
    Blue = 2
end
```

**Current State**: ❌ Enums not extracted to Julia bindings

### Arrays (2 types)
```cpp
struct Matrix3x3 {
    double data[9];  // 1D array
};

struct Grid {
    int cells[4][4];   // 2D array → NTuple{16, Cint}
    double values[3];  // 1D array → NTuple{3, Cdouble}
};
```

**Expected Julia:**
```julia
mutable struct Matrix3x3
    data::NTuple{9, Cdouble}
end

mutable struct Grid
    cells::NTuple{16, Cint}      # Flattened 4x4 = 16
    values::NTuple{3, Cdouble}
end
```

**Current State**: ⚠️ Arrays may not be sized correctly

### Function Pointers
```cpp
typedef int (*IntCallback)(double x, double y);

struct ComplexType {
    IntCallback handler;  // Function pointer member
};

// Function taking function pointer
int apply_callback(IntCallback cb, double x, double y);
```

**Expected Julia:**
```julia
mutable struct ComplexType
    handler::Ptr{Cvoid}  # Or better: typed function pointer
end

function apply_callback(cb::Ptr{Cvoid}, x::Cdouble, y::Cdouble)::Cint
    ccall((:apply_callback, LIB), Cint, (Ptr{Cvoid}, Cdouble, Cdouble), cb, x, y)
end
```

**Current State**: ❌ Function pointers likely not detected

---

## Issue Priority Matrix

| Issue | Impact | Difficulty | Priority |
|-------|---------|-----------|----------|
| **Enum extraction** | HIGH | MEDIUM | 🔥 P0 |
| **Parameter types** | HIGH | LOW | 🔥 P0 |
| **Array dimensions** | HIGH | MEDIUM | 🔥 P0 |
| **Return types** | HIGH | LOW | 🔥 P0 |
| **Function pointers** | MEDIUM | HIGH | P1 |
| **Struct member types** | HIGH | LOW | 🔥 P0 |
| **Const correctness** | LOW | LOW | P2 |

---

## P0 Issues (Fix First)

### 1. Enum Extraction & Mapping

**Problem**: C++ enums not appearing in Julia bindings

**Investigation needed:**
- Does `extract_dwarf_return_types()` find enums in DWARF?
- Are they stored in `struct_definitions`?
- Does Wrapper.jl generate `@enum` declarations?

**Files to check:**
- `src/Compiler.jl` - DWARF enum extraction (~line 800+)
- `src/Wrapper.jl` - `@enum` code generation (~line 1100+)

**Test:**
```julia
# After fixing, should generate:
@enum Color::Cuint begin
    Red = 0
    Green = 1
    Blue = 2
end

# And function should use it:
function color_to_int(c::Color)::Cint
    ccall((:color_to_int, LIB), Cint, (Color,), c)
end
```

### 2. Parameter Type Accuracy

**Problem**: Functions have parameters but types are generic `Any`

**Investigation needed:**
- Are parameters extracted from DWARF? (YES - we confirmed this)
- Are parameter types mapped correctly? (C++ `Matrix3x3` → Julia `Matrix3x3`)
- Is Wrapper.jl using the parameter types?

**Files to check:**
- `src/Compiler.jl` - Parameter extraction from DWARF
- `src/Wrapper.jl` - Parameter usage in function generation

**Current behavior:**
```julia
# BAD (current):
function matrix_sum(args...)
    ccall((:matrix_sum, LIB), Any, (), args...)
end

# GOOD (target):
function matrix_sum(m::Matrix3x3)::Cdouble
    ccall((:matrix_sum, LIB), Cdouble, (Matrix3x3,), m)
end
```

### 3. Return Type Accuracy

**Problem**: Functions returning structs/enums show `Any` return type

**Investigation needed:**
- Does DWARF have return type info? (YES - confirmed)
- Is return type mapping working? (`double` → `Cdouble` works, but `Color` → `Any`)
- Are custom types (enums, structs) in scope when generating wrappers?

**Files to check:**
- `src/Compiler.jl` - Return type extraction
- `src/Wrapper.jl` - Return type usage in function generation

### 4. Struct Member Type Accuracy

**Problem**: Struct members showing as `Any` instead of concrete types

**Example issue:**
```julia
# BAD (current):
mutable struct ComplexType
    color::Any       # Should be Color (enum)
    status::Any      # Should be Status (enum)
    handler::Any     # Should be Ptr{Cvoid}
end

# GOOD (target):
mutable struct ComplexType
    color::Color
    status::Status
    handler::Ptr{Cvoid}
end
```

**Investigation needed:**
- Are struct members extracted with type info?
- Is type mapping working for struct members?
- Are referenced types (enums) defined before structs that use them?

---

## P1 Issues (Fix After P0)

### 5. Function Pointer Support

**Problem**: Function pointers not detected or typed

**Challenges:**
- DWARF may represent function pointers as addresses
- Need to detect `(*funcname)(args)` pattern
- Julia representation: `Ptr{Cvoid}` or typed `Ptr{<FunctionType>}`

**Investigation needed:**
- Does DWARF include function pointer signatures?
- Can we extract parameter and return types for function pointers?
- How to represent in Julia? (Safe: `Ptr{Cvoid}`, Advanced: typed)

---

## Verification Strategy

### Test File Analysis Script

Create `test/verify_bindings_accuracy.jl`:
```julia
"""
Comprehensive binding accuracy verification.
Compares C++ source → DWARF metadata → Julia bindings.
"""
using RepliBuild
using Test

# 1. Compile test_advanced_types.cpp
# 2. Extract DWARF metadata
# 3. Generate Julia bindings
# 4. Verify accuracy:
#    - All enums present?
#    - All array dimensions correct?
#    - All function parameters typed?
#    - All return types accurate?
#    - All struct members typed?
```

### Manual Verification Checklist

After each fix, check:
- [ ] Compile `test_advanced_types.cpp` with RepliBuild
- [ ] Inspect generated Julia module
- [ ] Verify enums: `@enum Color`, `@enum Status`, `@enum Direction`
- [ ] Verify structs: `Matrix3x3`, `Grid`, `ComplexType` with correct member types
- [ ] Verify functions: All have typed parameters and return types
- [ ] Run `include("generated_module.jl")` - no syntax errors
- [ ] Call functions - no type errors

---

## Implementation Approach

### Phase 1: Deep Dive into Current DWARF Extraction
**Goal**: Understand exactly what data we currently extract

1. Add debug logging to `extract_dwarf_return_types()`
2. Compile test_advanced_types.cpp
3. Examine metadata JSON output
4. Document what's extracted vs what's missing

**Questions to answer:**
- Are enums in the metadata? Where?
- Are array dimensions preserved?
- Are function pointers detected?
- Are parameter types complete?

### Phase 2: Fix DWARF Extraction (if needed)
**Goal**: Ensure ALL type information is extracted

1. Enhance DWARF parser to extract:
   - Enum definitions with values
   - Array dimensions (not just element type)
   - Function pointer signatures
   - Complete struct member types

2. Store in metadata with proper structure

### Phase 3: Fix Type Mapping
**Goal**: Ensure C++ types → Julia types correctly

1. Build complete type mapping table
2. Handle custom types (enums, structs defined in the same project)
3. Ensure types are defined before use (topological sort)

### Phase 4: Fix Wrapper Generation
**Goal**: Use extracted metadata correctly

1. Generate `@enum` declarations
2. Generate struct definitions with typed members
3. Generate functions with typed parameters
4. Order definitions correctly (enums before structs before functions)

### Phase 5: Comprehensive Testing
**Goal**: Verify accuracy on real-world code

1. test_advanced_types.cpp (comprehensive edge cases)
2. Real C++ library (e.g., small header-only library)
3. Automated accuracy verification

---

## Success Metrics

### Must Have (P0)
- ✅ **100% enum extraction**: All enums in C++ appear as `@enum` in Julia
- ✅ **100% parameter accuracy**: All functions have correct typed parameters
- ✅ **100% return type accuracy**: All functions return correct types
- ✅ **100% struct member accuracy**: All struct fields have correct types
- ✅ **Array dimension accuracy**: All arrays have correct sizes

### Nice to Have (P1)
- ✅ **Function pointer support**: Detected and typed as `Ptr{Cvoid}` minimum
- ✅ **Demangled names**: Functions use clean names, not mangled symbols
- ✅ **Const correctness**: `const` preserved in signatures

### Future (P2)
- **Template support**: Handle C++ templates (hard)
- **Advanced function pointers**: Typed function pointers with signatures
- **Method support**: Class methods as standalone functions
- **Operator overloads**: Handle C++ operators

---

## Key Files Reference

| File | Purpose | Lines to Focus |
|------|---------|----------------|
| `src/Compiler.jl` | DWARF extraction, type extraction | 400-1000 |
| `src/Wrapper.jl` | Julia binding generation | 800-1400 |
| `test/test_advanced_types.cpp` | Comprehensive test case | All |
| `test/runtests.jl` | Current test suite | All |

---

## Next Steps

1. **Investigate current state**:
   - Run `test_advanced_types.cpp` through RepliBuild
   - Examine generated Julia bindings
   - Compare to C++ source
   - Document gaps

2. **Prioritize fixes**:
   - Start with enums (most visible, clear expected output)
   - Then parameters (already partially working)
   - Then array dimensions
   - Then function pointers

3. **Implement incrementally**:
   - Fix one issue at a time
   - Test after each fix
   - Don't break existing 27 tests

4. **Document as we go**:
   - Comments explaining DWARF structures
   - Examples of correct vs incorrect output
   - Type mapping tables

---

**Created**: November 25, 2025 (after modularization reset)
**Status**: Planning - ready to start P0 fixes
**Owner**: Focus on FFI accuracy before any architectural changes
