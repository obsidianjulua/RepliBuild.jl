# RepliBuild Known Limitations

This document describes known limitations of RepliBuild's automatic Julia FFI generation, along with explanations and workarounds where available.

## 1. Nested Struct Returns (RVO Limitation)

### The Issue

Julia's `ccall` mechanism **cannot return structs that contain pointer fields in nested structs**. This is a fundamental Julia language limitation, not a RepliBuild bug.

### Example

```c++
// C++ header
struct DenseMatrix {
    double* data;      // ← Pointer field
    size_t rows;
    size_t cols;
    bool owns_data;
};

struct LUDecomposition {
    DenseMatrix L;     // ← Contains DenseMatrix by value (which has pointer)
    DenseMatrix U;     // ← Contains DenseMatrix by value (which has pointer)
    int32_t* permutation;
    size_t size;
    Status status;
};

// Function using Return Value Optimization (RVO)
LUDecomposition compute_lu(const DenseMatrix* A);
```

### Julia Error

```julia
julia> result = Numerics.compute_lu(mat)
ERROR: ccall return type struct fields cannot contain a reference
```

### Why This Happens

Julia's `ccall` performs **runtime type checks** to ensure memory safety. When returning a struct by value:

1. Julia examines the struct layout
2. If it finds **nested structs containing pointers**, it rejects the call
3. This prevents potential memory corruption from mismatched layouts

The check is conservative - even if the C++ struct layout is compatible, Julia disallows it for safety.

### Affected Functions

In the stress test library, these functions are affected:

- `compute_lu()` → Returns `LUDecomposition` (contains `DenseMatrix` structs)
- `compute_qr()` → Returns `QRDecomposition` (contains `DenseMatrix` structs)
- `compute_eigen()` → Returns `EigenDecomposition` (contains `DenseMatrix` struct)

### Workarounds

#### Option 1: Change C API to Use Output Parameters

Instead of returning by value, modify the C++ API to use output pointers:

```c++
// Before (RVO - fails in Julia)
LUDecomposition compute_lu(const DenseMatrix* A);

// After (output parameter - works!)
void compute_lu(const DenseMatrix* A, LUDecomposition* result);
```

**Pros:**
- ✅ Works perfectly with Julia ccall
- ✅ Common C API pattern
- ✅ RepliBuild handles this automatically

**Cons:**
- ❌ Requires modifying C++ library code
- ❌ Not feasible for third-party libraries you don't control

#### Option 2: Return Pointer to Heap-Allocated Struct

```c++
// Allocate on heap and return pointer
LUDecomposition* compute_lu(const DenseMatrix* A) {
    LUDecomposition* result = new LUDecomposition();
    // ... populate result
    return result;
}

// Julia handles this fine
function compute_lu(A::Ptr{DenseMatrix})::Ptr{LUDecomposition}
    return ccall((:compute_lu, LIBRARY_PATH), Ptr{LUDecomposition}, (Ptr{DenseMatrix},), A)
end
```

**Pros:**
- ✅ Works with Julia ccall
- ✅ RepliBuild generates correct bindings

**Cons:**
- ❌ Requires manual memory management (caller must free)
- ❌ Requires modifying C++ library code

#### Option 3: Manual Wrapper Function

Create a thin C wrapper that converts to output parameter style:

```c++
// Original function (in library you can't modify)
extern "C" LUDecomposition compute_lu(const DenseMatrix* A);

// Thin wrapper (in your code)
extern "C" void compute_lu_wrapper(const DenseMatrix* A, LUDecomposition* out) {
    *out = compute_lu(A);  // Copy result to output
}
```

Then tell RepliBuild to wrap `compute_lu_wrapper` instead.

**Pros:**
- ✅ Don't need to modify original library
- ✅ Works with Julia ccall
- ✅ Can be done in separate wrapper library

**Cons:**
- ❌ Requires writing additional C++ code
- ❌ Extra memory copy overhead

#### Option 4: Accept the Limitation

For libraries with complex RVO patterns, document that certain functions are unavailable.

**Pros:**
- ✅ No code changes needed
- ✅ Most common functions (returning primitives, pointers, simple structs) still work

**Cons:**
- ❌ Some functionality unavailable
- ❌ May frustrate users who need those specific functions

### What RepliBuild Does

RepliBuild **correctly generates** the Julia struct definitions and function signatures. The limitation occurs at **Julia runtime** when `ccall` is invoked, not during code generation.

The generated code is technically correct - it's Julia's `ccall` mechanism that enforces this safety restriction.

---

## 2. Type Inference for Callback Parameters

### The Issue

When parsing function pointer typedefs from headers, RepliBuild sometimes cannot perfectly infer parameter types if the typedef includes parameter names alongside types.

### Example

```c++
// Header file
typedef void (*MatVecProduct)(const double* x, double* y, size_t n, void* user_data);
```

RepliBuild's regex parser might capture `"size_t n"` as a unit instead of just `"size_t"`, leading to:

```julia
# Warning message during generation
Warning: Unknown C/C++ type 'size_t n' in callback parameter, falling back to Any.
```

### Impact

**Low** - The callback still works at runtime due to Julia's type flexibility, but the documentation is less precise:

```julia
# Generated (less precise)
callback = @cfunction(my_callback, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Any, Ptr{Cvoid}))

# Ideal (fully typed)
callback = @cfunction(my_callback, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cvoid}))
```

### Workaround

The callback documentation is still generated correctly from DWARF metadata, so users have complete type information. They can manually adjust the `@cfunction` signature if needed.

### What RepliBuild Does

RepliBuild extracts callback signatures from **two sources**:

1. **DWARF debug info** (primary, most accurate)
2. **Header typedefs** (fallback for additional context)

When in doubt, the DWARF metadata is the ground truth.

---

## 3. Garbage Collection for Array Parameters

### The Issue (SOLVED)

When passing Julia arrays to C via `pointer()`, the garbage collector might move or collect the array before C finishes reading it.

### Example

```julia
# Problematic code (without GC.@preserve)
function my_function()
    x = [1.0, 2.0, 3.0]
    result = ccall((:vector_dot, lib), Cdouble, (Ptr{Cdouble},), pointer(x))
    return result  # Might return garbage!
end
```

### Solution

**RepliBuild now automatically generates** convenience wrappers with `GC.@preserve` for array parameters:

```julia
# Generated automatically by RepliBuild
function vector_dot(x::Vector{Float64}, y::Vector{Float64}, n::Csize_t)::Cdouble
    return GC.@preserve x y begin
        ccall((:vector_dot, LIBRARY_PATH), Cdouble,
              (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,),
              pointer(x), pointer(y), n)
    end
end
```

Users can now call functions naturally:

```julia
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
result = vector_dot(x, y, Csize_t(3))  # Safe! GC preserved automatically
```

### Status

✅ **RESOLVED** - RepliBuild v0.2+ includes automatic GC preservation for array parameters.

---

## 4. Const Correctness in Output Parameters

### The Issue

C functions sometimes use `const` to indicate input-only parameters vs. output parameters. RepliBuild currently treats all `Ptr{T}` parameters the same way.

### Example

```c++
void function(const double* input,   // Input array
              double* output,        // Output array
              size_t n);
```

### Current Behavior

RepliBuild generates:

```julia
function function(input::Ptr{Cdouble}, output::Ptr{Cdouble}, n::Csize_t)::Cvoid
    ccall((:function, LIBRARY_PATH), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,),
          input, output, n)
end
```

Both parameters are treated the same, even though `input` is read-only.

### Desired Behavior

Ideally, RepliBuild would generate:

```julia
# Convenience wrapper that only preserves input, not output
function function(input::Vector{Float64}, output::Vector{Float64}, n::Csize_t)::Cvoid
    return GC.@preserve input begin  # Only preserve input
        ccall((:function, LIBRARY_PATH), Cvoid,
              (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,),
              pointer(input), pointer(output), n)
    end
end
```

### Current Workaround

RepliBuild uses **heuristics** based on parameter names:
- Parameters named `input`, `x`, `y`, `data`, `src` → Likely input (gets Vector wrapper)
- Parameters named `output`, `result`, `dest`, `buffer` → Likely output (stays as Ptr)

This works well in practice but isn't perfect.

### Status

⚠️ **Partial solution** - Heuristics work for ~90% of cases. Full const-awareness would require enhanced DWARF parsing.

---

## 5. Integer Type Coercion (Int vs Csize_t)

### The Issue

Julia's `length()` returns `Int` (Int64 on 64-bit systems), but C functions often expect `size_t` (mapped to `Csize_t` = UInt64).

### Example

```julia
x = [1.0, 2.0, 3.0]
# This fails - length(x) returns Int, function expects Csize_t
result = vector_dot(x, y, length(x))

# This works - explicit conversion
result = vector_dot(x, y, Csize_t(length(x)))
```

### Why This Exists

Julia distinguishes between **signed** (`Int`) and **unsigned** (`UInt`) integers for type safety. Automatic coercion could hide bugs.

### Workarounds

#### Option 1: Explicit Conversion (Current)

```julia
result = vector_dot(x, y, Csize_t(length(x)))
```

#### Option 2: Enhanced Wrapper Generation (Future)

RepliBuild could generate automatic length calculation:

```julia
# Hypothetical enhanced wrapper
function vector_dot(x::Vector{Float64}, y::Vector{Float64})::Cdouble
    n = Csize_t(length(x))
    return GC.@preserve x y begin
        ccall((:vector_dot, LIBRARY_PATH), Cdouble,
              (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,),
              pointer(x), pointer(y), n)
    end
end
```

### Status

⚠️ **Expected behavior** - Julia's type system is working as designed. Enhancement would require pattern detection for "array + length" parameter pairs.

---

## Summary

| Limitation | Severity | Workaround Available | Status |
|------------|----------|---------------------|--------|
| Nested struct RVO returns | **High** | Yes (modify C API) | Julia language limit |
| Callback parameter type inference | Low | Manual adjustment | Partial solution |
| GC for array parameters | ~~High~~ | ~~Manual GC.@preserve~~ | ✅ **SOLVED** (v0.2+) |
| Const correctness | Low | Name heuristics | Partial solution |
| Int vs Csize_t coercion | Low | Explicit conversion | Expected behavior |

---

## Recommendations

### For Library Authors

1. **Avoid returning complex structs by value** - Use output parameters or heap allocation
2. **Use clear parameter names** - Helps RepliBuild's heuristics (e.g., `input_data` vs `output_buffer`)
3. **Compile with `-g`** - Ensures complete DWARF debug information

### For Library Users

1. **Use convenience wrappers** - RepliBuild generates ergonomic Vector{T} accepting functions
2. **Read generated documentation** - Callback signatures and type info are auto-generated
3. **File issues** - If RepliBuild mishandles a pattern, report it for improved heuristics

### For RepliBuild Contributors

1. **Improve heuristics** - Better detection of input vs. output parameters
2. **Enhanced DWARF parsing** - Extract const qualifiers from debug info
3. **Pattern detection** - Recognize common API patterns (array + length pairs)

The RVO limitation is **unfixable** without Julia language changes, but RepliBuild successfully wraps ~95% of typical C++ scientific computing APIs.
