# DWARF Extraction Findings & Recommendations

## Current Coverage: 78.0% (940/1206 instances)
**Previous**: 71.9% → **Improved**: +6.1%

RepliBuild extracts the majority of DWARF information but is missing some key features.

## High-Impact Missing Features

### 1. **typedef** (63 instances, 5.2%) ✅ EXTRACTED, 🔧 RESOLUTION PENDING
**Status**: Extracted but not fully resolved
**Impact**: Type aliases extracted but not chained to final types
**What works**: Typedef entries captured in DWARF metadata
**What's missing**: Resolution chain (int32_t → __int32_t → int → Cint)
**Action**: Implement typedef resolution in type mapping

### 2. **imported_declaration** (221 instances, 18.3%) 🎯
**Status**: Not extracted
**Impact**: Using declarations are lost (std namespace imports, etc.)
**Example**: `using std::string;`
**Action**: Medium priority - mainly affects namespace clarity

### 3. **template_type_parameter** (2 instances) ✅ EXTRACTED
**Status**: Fully extracted
**Impact**: Template instantiation info captured
**Metadata**: `"templates": [{"kind": "type", "name": "T", "type": "int"}]`
**Wrapper**: Template names sanitized (`Pair<int>` → `Pair_int`)

### 4. **template_value_parameter** (1 instance) ✅ EXTRACTED
**Status**: Fully extracted
**Impact**: Template value parameters captured
**Metadata**: `"templates": [{"kind": "value", "name": "N", "value": 10}]`
**Wrapper**: Names sanitized (`FixedArray<float, 10>` → `FixedArray_float_10`)

### 5. **inheritance** (2 instances) ✅ EXTRACTED
**Status**: Fully extracted
**Impact**: Class hierarchies captured in metadata
**Metadata**: `"inherits_from": ["Shape"]` with accessibility info
**Wrapper**: Available in JSON for documentation (Julia doesn't support inheritance)

### 6. **namespace** (6 instances) ✅ EXTRACTED
**Status**: Fully extracted
**Impact**: Namespace information captured
**Wrapper**: Functions name-mangled (`math::pi()` → `math_pi()`)

## Lower Priority

### **unspecified_parameters** (15 instances, 1.2%)
Marks variadic functions (`...`). Currently not detected.

### **restrict_type** (15 instances, 1.2%)
C99 `restrict` qualifier - rarely used in C++.

### **variable** (11 instances, 0.9%)
Global variables - could be useful for wrapping.

### **union_type** (1 instance)
Unions - currently treated as structs (may have layout issues).

### **volatile_type** (1 instance)
Volatile qualifier - currently stripped.

## What We're Doing Well

✅ **Functions** - 252 subprograms extracted (239 with signatures)
✅ **Parameters** - 457 formal_parameter tags
✅ **Primitives** - 17 base_type tags
✅ **Pointers** - 49 pointer_type tags
✅ **Structs** - 15 structure_type + 4 class_type (9 extracted with members)
✅ **Enums** - 3 enumeration_type (3 extracted with values)
✅ **Arrays** - 3 array_type with subrange

## Implementation Status

✅ **COMPLETED**:
1. ✅ **template_type_parameter** - Template types tracked
2. ✅ **template_value_parameter** - Template values tracked
3. ✅ **inheritance** - Class hierarchies captured
4. ✅ **namespace** - Namespace-aware function names
5. ✅ **typedef extraction** - Typedefs captured in DWARF

🔧 **IN PROGRESS**:
1. 🔧 **typedef resolution** - Type chain resolution needed

📋 **REMAINING**:
1. **unspecified_parameters** - Variadic function detection (15 instances)
2. **imported_declaration** - Namespace import tracking (221 instances)
3. **variable** - Global variable wrapping (11 instances)
4. **union_type** - Union handling (1 instance)
5. **volatile_type** - Volatile qualifier (1 instance)
6. **restrict_type** - Restrict qualifier (15 instances)

## Tools Comparison

**dwarfdump vs readelf**:
- dwarfdump: 267KB output, more readable format
- readelf: 258KB output, standard tool (always available)
- objdump: 258KB output, similar to readelf

**Recommendation**: Use readelf (current choice) - it's always available and provides all needed info.

## Files for Deep Analysis

- `dwarfdump_full.txt` - Human-readable DWARF dump (267KB)
- `readelf_full.txt` - Standard DWARF dump (253KB)
- `replibuild_extraction.json` - What RepliBuild extracted (40KB)
- `tag_comparison.txt` - Tag frequency analysis

## Next Steps

1. Review `dwarfdump_full.txt` for typedef patterns
2. Implement typedef tracking in [Compiler.jl:extract_dwarf_return_types](../src/Compiler.jl#L643)
3. Add typedef resolution to type registry in Wrapper.jl
4. Re-run test to verify improvement
5. Move on to templates and inheritance
