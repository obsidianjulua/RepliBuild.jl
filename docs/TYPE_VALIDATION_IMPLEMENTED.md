# Type Validation System - Implementation Complete ✅

**Date**: November 25, 2025
**Status**: ✅ Fully Implemented and Tested

---

## Summary

Successfully implemented a comprehensive type validation and error handling system for RepliBuild's C/C++ to Julia FFI type mapping. The system provides three strictness modes with smart heuristics and context-aware error messages.

---

## What Was Implemented

### 1. Type Strictness Enum (`TypeStrictness`)

Three validation modes to control how unknown/unmapped types are handled:

```julia
@enum TypeStrictness begin
    STRICT      # Error on unmapped types (recommended for production)
    WARN        # Warn and fallback (useful for debugging)
    PERMISSIVE  # Silent fallback (legacy compatibility)
end
```

### 2. Enhanced TypeRegistry

Added validation settings to the `TypeRegistry` struct:

```julia
struct TypeRegistry
    # ... existing fields ...

    # NEW: Type validation settings
    strictness::TypeStrictness           # How to handle unknown types
    allow_unknown_structs::Bool          # Treat unknown types as opaque structs
    allow_unknown_enums::Bool            # Treat unknown enums as Cint
    allow_function_pointers::Bool        # Map function pointers to Ptr{Cvoid}
end
```

**Default Settings:**
- `strictness = WARN` (warnings but no errors)
- `allow_unknown_structs = true` (treats struct-like names as opaque types)
- `allow_unknown_enums = false` (enums must be explicitly mapped)
- `allow_function_pointers = true` (maps to `Ptr{Cvoid}`)

### 3. Type Heuristics

Smart detection functions to categorize unknown types:

- **`is_struct_like()`**: Detects struct/class names (e.g., `Matrix3x3`, `Grid`)
- **`is_enum_like()`**: Detects enum names (same heuristics as structs for now)
- **`is_function_pointer_like()`**: Detects function pointer syntax `(*name)(...)`

### 4. Smart Error Handling

The `handle_unknown_type()` function provides intelligent fallbacks:

**STRICT Mode:**
```julia
ERROR: Unknown C/C++ type: 'Matrix3x3'
Context: parameter 1 of function matrix_sum

Suggestions:
1. Add custom type mapping
2. If this is a struct: ensure it's defined in headers with -g flag
3. If this is an enum: verify extraction is working
4. If this is a function pointer: enable allow_function_pointers=true
```

**WARN Mode:**
```
Warning: Treating unknown type 'Matrix3x3' as opaque struct in parameter 1
```

**PERMISSIVE Mode:**
Silent fallback to `Any`

### 5. Context-Aware Error Messages

Type inference now passes context through the entire call chain:

```julia
infer_julia_type(registry, "BadType", context="parameter 1 of function foo")
# Error includes: "parameter 1 of function foo"

infer_julia_type(registry, "UnknownElem*", context="return type")
# Error includes: "return type (pointer to UnknownElem)"
```

---

## Usage Examples

### Example 1: STRICT Mode (Recommended)

```julia
using RepliBuild
using RepliBuild.Wrapper: create_type_registry, STRICT

# Load config
config = load_config("replibuild.toml")

# Create strict registry
registry = create_type_registry(config,
    strictness=STRICT,
    allow_unknown_structs=false)

# Will error on unknown types with helpful suggestions
try
    infer_julia_type(registry, "UnknownStruct")
catch e
    # Shows detailed error with suggestions
end
```

### Example 2: WARN Mode (Development)

```julia
# Allow unknown structs with warnings
registry = create_type_registry(config,
    strictness=WARN,
    allow_unknown_structs=true)

# Will warn and use C++ name as Julia type
julia_type = infer_julia_type(registry, "MyStruct")
# => "MyStruct" (with warning logged)
```

### Example 3: Custom Type Mappings

```julia
custom_types = Dict(
    "Matrix3x3" => "Matrix3x3",      # Explicitly map custom struct
    "ErrorCode" => "Cint",            # Map enum to Cint
    "Handle" => "Ptr{Cvoid}"          # Map handle to pointer
)

registry = create_type_registry(config,
    custom_types=custom_types,
    strictness=STRICT)

# Now these types work in STRICT mode
@test infer_julia_type(registry, "Matrix3x3") == "Matrix3x3"
@test infer_julia_type(registry, "ErrorCode") == "Cint"
```

---

## Files Modified

### Core Implementation
- **[src/Wrapper.jl](../src/Wrapper.jl)**
  - Added `TypeStrictness` enum (lines 31-35)
  - Enhanced `TypeRegistry` struct (lines 57-62)
  - Added `create_type_registry()` parameters (lines 81-86)
  - Implemented type heuristics (lines 205-243)
  - Implemented `handle_unknown_type()` (lines 264-355)
  - Updated `infer_julia_type()` with context parameter (line 388)
  - All recursive calls now pass context through
  - All fallback "Any" returns replaced with `handle_unknown_type()`

### Tests
- **[test/test_type_validation.jl](../test/test_type_validation.jl)** (NEW)
  - 66 comprehensive tests covering all modes and edge cases
  - Tests for type heuristics
  - Tests for each strictness mode
  - Tests for context-aware error messages
  - Tests for all known type mappings
  - Tests for custom type mappings

### Documentation
- **[docs/TYPE_VALIDATION_PLAN.md](TYPE_VALIDATION_PLAN.md)** (Design doc)
- **[docs/TYPE_VALIDATION_IMPLEMENTED.md](TYPE_VALIDATION_IMPLEMENTED.md)** (This file)

---

## Test Results

```
Test Summary:   | Pass  Total  Time
Type Heuristics |   17     17  0.5s
Type Strictness Modes |   16     16  2.0s
Context-Aware Error Messages |    2      2  0.0s
Known Type Mappings |   26     26  0.1s
Custom Type Mappings |    5      5  0.1s

✓ All type validation tests passed!
```

**Total**: 66/66 tests passing

---

## Backwards Compatibility

✅ **Fully Backwards Compatible**

- Default mode is `WARN` (not `STRICT`)
- `allow_unknown_structs=true` by default (treats struct-like names as opaque types)
- Existing code continues to work unchanged
- Users can opt-in to stricter validation

---

## Next Steps

Now that type validation is solid, the next priorities are:

### P0: Enum Extraction
- Extract enums from DWARF metadata
- Generate `@enum` declarations in Julia bindings
- Map enum types correctly in function signatures

### P1: Array Dimensions
- Handle multi-dimensional arrays correctly
- Flatten `int[4][4]` → `NTuple{16, Cint}`

### P2: Function Pointers
- Improve detection beyond heuristics
- Extract function pointer signatures from DWARF
- Consider typed function pointers vs `Ptr{Cvoid}`

### P3: Parameter & Return Types
- Use validation system when generating wrappers
- Pass context ("parameter 1 of function X") to `infer_julia_type()`
- Ensure all generated bindings have proper types (no generic `Any`)

---

## Configuration Recommendations

### For Production (Strict Validation)

```julia
registry = create_type_registry(config,
    strictness=STRICT,
    allow_unknown_structs=true,   # Allow user-defined structs
    allow_unknown_enums=false,     # Enums must be extracted
    allow_function_pointers=true)  # Allow callbacks
```

### For Development (Permissive with Warnings)

```julia
registry = create_type_registry(config,
    strictness=WARN,
    allow_unknown_structs=true,
    allow_unknown_enums=true,      # More lenient during dev
    allow_function_pointers=true)
```

### For Legacy Code (Maximum Compatibility)

```julia
registry = create_type_registry(config,
    strictness=PERMISSIVE,
    allow_unknown_structs=true,
    allow_unknown_enums=true,
    allow_function_pointers=true)
```

---

## Benefits

### Before This Implementation

```julia
# Generated (BROKEN):
function matrix_sum(args...)::Any
    ccall((:matrix_sum, LIB), Any, (), args...)
end
```

Problems:
- Runtime type errors
- No indication of what went wrong
- Hard to debug
- Silent failures

### After This Implementation

#### In WARN mode (default):
```
Warning: Treating unknown type 'Matrix3x3' as opaque struct in parameter 1 of function matrix_sum
```

```julia
# Generated (BETTER):
function matrix_sum(m::Matrix3x3)::Cdouble
    ccall((:matrix_sum, LIB), Cdouble, (Matrix3x3,), m)
end
```

#### In STRICT mode:
```
ERROR: Unknown C/C++ type: 'Matrix3x3'
Context: parameter 1 of function matrix_sum

[Detailed suggestions on how to fix...]
```

Forces you to fix the issue before generating bindings!

---

## Key Design Decisions

1. **Default to WARN, not STRICT**: Backwards compatibility and gradual migration
2. **allow_unknown_structs=true by default**: Most struct names can be used directly
3. **Context throughout**: Every type inference includes where it's being used
4. **Smart heuristics**: Detect struct/enum/function-pointer patterns automatically
5. **Helpful errors**: Multi-line error messages with specific suggestions
6. **Extensible**: Easy to add new type categories and validation rules

---

## Performance

- **Negligible overhead**: Validation only runs during binding generation (one-time cost)
- **No runtime cost**: Generated bindings are identical to before (when types are known)
- **Fast heuristics**: Regex patterns are simple and compile once

---

## Future Enhancements

1. **DWARF-based enum detection**: Check against extracted enums instead of heuristics
2. **Template registry**: Allow registering common template patterns
3. **Configuration file support**: Define strictness in `replibuild.toml`
4. **IDE integration**: LSP could show type validation warnings in real-time
5. **Type confidence scoring**: Track which types are "certain" vs "inferred"

---

**Status**: ✅ Ready for production use
**Recommended**: Use WARN mode initially, migrate to STRICT for mature projects

