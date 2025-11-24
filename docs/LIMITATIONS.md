# RepliBuild Limitations

This document defines the correctness boundaries and rejection rules for RepliBuild's FFI generation.

## Technical Position

**RepliBuild's Status:** First Julia system (and one of the first in any language) to combine three metadata sources for automatic C++ FFI:

1. **DWARF debug information** - Semantic types from compilation (DW_TAG_* DIEs)
2. **LLVM IR** - Canonical ABI layouts and calling conventions
3. **Symbol tables** - Function signatures (mangled/demangled)

**Comparison to existing approaches:**

| Tool | Language | Metadata Source | C++ Templates | Manual Work | Constraints |
|------|----------|----------------|---------------|-------------|-------------|
| **RepliBuild** | Julia | DWARF + IR + symbols | Instantiated only | Zero | Standard-layout types |
| **DragonFFI** | Python | DWARF + IR | N/A (C only) | Zero | C types only |
| **Clang.jl** | Julia | Source AST | Limited | Medium | Requires headers |
| **CxxWrap.jl** | Julia | Manual annotations | Full | High | Developer writes all |
| **pybind11** | Python | C++ annotations | Full | High | Requires C++ code |

**Innovation:** Extends DragonFFI's DWARF + IR approach (pioneering work for C) to C++ with template instantiation support, struct member extraction, and three-way cross-validation for ABI correctness.

**Acknowledgment:** DragonFFI pioneered DWARF + IR for automatic FFI (C). RepliBuild builds on this foundation for C++ and Julia.

**Validation:** Approach confirmed as novel for Julia by JuliaHub developers.

---

## Supported Types

### Guaranteed Correct
These types can be safely wrapped with correct ABI semantics:

- **POD types** (Plain Old Data)
  - C structs
  - C++ standard-layout structs without constructors/destructors
  - Trivially-copyable types

- **Base types**
  - Signed/unsigned integers (char, short, int, long, long long)
  - Fixed-width integers (int8_t, int16_t, int32_t, int64_t, etc.)
  - Floating point (float, double)
  - bool, void

- **Pointers and references**
  - T* (pointers to any type)
  - const T* (pointers to const)
  - T& (references, passed as pointers)

- **Functions**
  - Functions taking/returning supported types
  - Extern "C" linkage or mangled C++ names

### Template Instantiations
- **Supported:** Template instantiations that are ODR-used and appear in DWARF
- **Not supported:** Templates not instantiated in the compiled code
- **Limitation:** Cannot force instantiation; must be used in compiled code

## Unsupported / Rejected Types

### C++ Object-Oriented Features
RepliBuild **rejects** these features:

- **Virtual methods and vtables**
  - Reason: ABI-unsafe, layout compiler-dependent
  - Impact: Classes with virtual functions cannot be wrapped

- **Inheritance**
  - Reason: Base class layout, virtual inheritance, access control
  - Impact: Derived classes rejected

- **Member function pointers**
  - Reason: Calling convention complexity, thunk generation
  - Impact: Cannot wrap method pointers

- **Operator overloading**
  - Reason: Name mangling varies, cannot guarantee resolution
  - Impact: Use explicit named functions instead

### STL Containers
STL containers are **fundamentally unsafe** to wrap:

- **std::vector, std::string, std::map, etc.**
  - Reason: Implementation-defined layout
  - Varies between: libstdc++ vs libc++, debug vs release, versions
  - Small string optimization (SSO) changes layout
  - Allocator model differences
  - Cannot guarantee ABI stability

- **Recommendation:** Use C-compatible APIs or pass POD types only

### Exception Specifications
- Functions marked `noexcept` or with exception specs are **not validated**
- RepliBuild assumes no exceptions cross FFI boundary
- Undefined behavior if C++ throws into Julia

### Types Optimized Out
DWARF may omit:
- Dead fields under aggressive optimization (-O2, -O3)
- Unused template instantiations
- Inlined function boundaries
- Types with no debug locations

**If not in DWARF, cannot be extracted.**

## ABI Assumptions

RepliBuild makes the following ABI assumptions:

### Platform
- **x86_64 Linux** (System V AMD64 ABI)
- **Clang/GCC** struct layout rules
- Other platforms: experimental, not validated

### Struct Layout
- Assumes DWARF accurately reflects compiled layout
- Assumes no padding removal under LTO
- Assumes alignment rules match System V ABI
- **Does not support:** `#pragma pack`, `__attribute__((packed))`

### Calling Convention
- Assumes standard C calling convention
- Struct-by-value follows System V rules (registers for small structs)
- Large structs passed by hidden pointer (ABI-dependent)

### Compiler Versions
- Validated: Clang 18-21, GCC 11-13
- **Risk:** ABI changes between compiler versions
- **Mitigation:** Recompile bindings with matching compiler

## Language Interoperability

### Rust
- **Supported:** `#[repr(C)]` types only
- **Not supported:** Rust native layouts, trait objects
- **Requirement:** Must use `extern "C"` functions

### Swift
- **Supported:** `@convention(c)` functions and C-compatible types
- **Not supported:** Swift native classes, protocols
- **Requirement:** Must use C bridging headers

### Fortran
- **Supported:** Types with `BIND(C)` attribute
- **Not supported:** Fortran native arrays, derived types
- **Requirement:** ISO_C_BINDING module required

**DWARF presence ≠ ABI safety.** Even if DWARF contains the type, it must be C-compatible.

## Rejection Rules

RepliBuild will **reject** wrapping when it detects:

1. **Virtual methods** (DW_TAG_subprogram with DW_AT_virtuality)
2. **Inheritance** (DW_TAG_inheritance present)
3. **Non-standard layout** (DW_AT_byte_size inconsistent with member sum)
4. **Incomplete DWARF** (missing DW_TAG_member for known fields)
5. **Unknown calling convention** (non-standard linkage)
6. **Template types without instantiation** (forward declarations only)
7. **Anonymous types** (lambda closures, local classes)

## Future Work

### Phase 7: Enums and Arrays
- DW_TAG_enumeration_type: Extractable, needs Julia mapping
- Fixed-size arrays: Can be wrapped as NTuple
- Dynamic arrays: Cannot safely wrap (size unknown)

### Phase 8: STL (Experimental)
- **High risk:** ABI instability
- **Approach:** Vendor-specific layouts, version-pinned
- **Not recommended** for production use

### Phase 9: Validation Tools
- DWARF-IR cross-validator (detect layout mismatches)
- ABI compatibility checker (compiler version drift)
- Runtime layout verification (assert sizeof matches)

## Correctness Guarantees

### What RepliBuild Guarantees
- Types extracted from DWARF match compiler's view at compilation time
- Generated ccall signatures match LLVM IR ABI
- Struct layouts for standard-layout types are correct

### What RepliBuild Does NOT Guarantee
- Types present in DWARF are all types in program
- DWARF layout matches runtime layout under all optimizations
- ABI stability across compiler versions
- Correctness for non-standard-layout types
- Exception safety

## Recommendations

### For Maximum Safety
1. Use C-compatible APIs (`extern "C"`)
2. Use POD types only (no constructors/destructors)
3. Compile with `-g -O0` (disable optimization for DWARF completeness)
4. Test with same compiler version for Julia and C++
5. Validate struct layouts with static_assert in C++

### For Template-Heavy Code
1. Explicitly instantiate templates you need
2. Verify instantiation appears in DWARF (readelf check)
3. Use explicit specialization over SFINAE
4. Avoid templated member functions (may not appear in DWARF)

### For Cross-Language Use
1. Define C-compatible interface layer
2. Use `#[repr(C)]` (Rust), `@convention(c)` (Swift), `BIND(C)` (Fortran)
3. Test ABI compatibility with layout assertions
4. Document compiler/version requirements

## Known Issues

1. **LTO may remove padding:** Link-time optimization can alter struct layout
2. **Debug vs Release layouts:** Some types differ between -O0 and -O2
3. **Compiler-specific extensions:** `__attribute__` may not be in DWARF
4. **Forward declarations:** Cannot extract types with incomplete DWARF

For questions about specific types, consult the source or file an issue.
