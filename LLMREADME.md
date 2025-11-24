# LLMREADME.md

This file provides guidance to LLM when working with code in this repository.

## Project Overview

**RepliBuild.jl** generates Julia FFI bindings for C/C++ libraries by extracting type information from DWARF debug data. It targets **standard-layout, trivially-copyable types**.

**Novel for Julia:** First Julia system (and one of the first in any language) to combine three metadata sources:
1. **DWARF debug information** (DW_TAG_* DIEs) - semantic types from compilation
2. **LLVM IR** - canonical ABI struct layouts
3. **Symbol tables** - function signatures and linkage

**Position in ecosystem:**
- **DragonFFI** (Python): Pioneered DWARF + IR for C (prior art we build on)
- **Clang.jl** (Julia): Source AST only, limited template support
- **CxxWrap.jl** (Julia): Manual annotations, high maintenance
- **RepliBuild** (Julia): Extends DWARF + IR to C++ templates, automatic

This three-way approach enables automatic C++ FFI without headers, for standard-layout types.

**Current State (v1.1.0):**
-  DWARF extraction: base types, pointers, const, references, structs
-  Standard-layout struct support with member extraction
-  Function signature extraction (parameters and return types)
-  Validated on Eigen (20K+ type DIEs extracted)
-  **Supports only:** POD types, standard-layout structs, C functions

**Correctness Boundary:**
- **Guaranteed:** Types in DWARF that are standard-layout and trivially-copyable
- **Not supported:** Virtual methods, inheritance, STL containers, non-standard-layout types
- See [LIMITATIONS.md](LIMITATIONS.md) for complete rejection rules

## Development Commands

### Simple 3-Command API

**RepliBuild has EXACTLY 3 commands you need:**

```julia
using RepliBuild

# 1. Compile C++ code → library
RepliBuild.build()

# 2. Generate Julia wrapper
RepliBuild.wrap()

# 3. Check status
RepliBuild.info()
```

That's it! Everything else is internal.

### Complete Example
```bash
# Test with the test project
cd test_cpp_project/

# In Julia:
julia --project=..
```

```julia
using RepliBuild

# Build C++ library
RepliBuild.build()   # → julia/libmathlib.so + metadata

# Generate Julia wrapper
RepliBuild.wrap()    # → julia/Mathlib.jl

# Use it!
include("julia/Mathlib.jl")
using .Mathlib
result = add(5, 3)  # Call C++ function from Julia
```

### Utility Commands
```julia
# Clean build artifacts
RepliBuild.clean()

# For advanced users only
RepliBuild.Compiler.compile_to_ir(config, files)
RepliBuild.Wrapper.wrap_library(config, lib_path)
RepliBuild.Discovery.discover(".", force=true)
```

## Architecture

### Core Pipeline
```
C++ Source → Compile with -g → DWARF Debug Info → Extract Types → Generate Julia Bindings
```

### Module Structure (12 files, 10 modules + 2 utilities)

**Actual Source Files:**
```
src/
├── RepliBuild.jl           # Main module with 3-command API (build, wrap, info)
├── RepliBuildPaths.jl      # Module: Path management
├── LLVMEnvironment.jl      # Module: LLVM toolchain detection
├── ConfigurationManager.jl # Module: Config loading (replibuild.toml)
├── BuildBridge.jl          # Module: External command execution
├── ASTWalker.jl            # Module: C++ dependency parsing
├── Discovery.jl            # Module: Project structure scanning
├── CMakeParser.jl          # Module: CMake import
├── ClangJLBridge.jl        # Module: Clang.jl integration
├── Compiler.jl             # Module: C++ → IR → binary + DWARF extraction
├── Wrapper.jl              # Module: Julia binding generation
└── WorkspaceBuilder.jl     # Module: Multi-library builds
```

**Module Roles:**

*Infrastructure (4 modules):*
- `RepliBuildPaths`: File path management and project structure
- `LLVMEnvironment`: LLVM toolchain detection and configuration
- `ConfigurationManager`: Load and parse replibuild.toml configs
- `BuildBridge`: Execute external commands (clang++, llvm-link, etc.)

*Core Build (5 modules):*
- `ASTWalker`: Parse C++ source to extract dependencies
- `Discovery`: Scan projects, find sources, generate configs
- `CMakeParser`: Import CMake projects to replibuild.toml
- `ClangJLBridge`: Integration with Clang.jl for header-based wrapping
- `Compiler`: **THE CORE INNOVATION** - C++ → LLVM IR → binary compilation + DWARF type extraction

*Wrapper Generation (1 module):*
- `Wrapper`: Generate type-safe Julia bindings using extracted metadata (3-tier system)

*Build Orchestration (1 module):*
- `WorkspaceBuilder`: Multi-library workspace builds with dependency ordering

*User Interface:*
- Simple 3-command API in main module: `build()`, `wrap()`, `info()`

**Historical Note:** Bridge_LLVM.jl was removed and replaced by Compiler.jl during Phase 1 cleanup. If you see references to Bridge_LLVM, they are outdated.

### DWARF Extraction Process (The Core Innovation)

Location: [Compiler.jl:475-900](src/Compiler.jl#L475-L900), function `extract_dwarf_return_types()`

**What We Actually Do:**

1. **Compile with debug info** (`-g` flag) - CRITICAL, this generates DWARF data
2. **Extract DWARF using readelf** `--debug-dump=info` to get raw debug information
3. **Parse DWARF tags** (the compiler's own type database):
   - `DW_TAG_base_type` → int, double, bool, char ( Working)
   - `DW_TAG_pointer_type` → T* ( Working)
   - `DW_TAG_const_type` → const T ( Working)
   - `DW_TAG_reference_type` → T& ( Working)
   - `DW_TAG_structure_type` → struct definitions with members ( Working)
   - `DW_TAG_class_type` → class definitions ( Detection works)
   - `DW_TAG_subprogram` → function signatures ( Working)
   - `DW_TAG_member` → struct/class member fields with offsets ( Working)
4. **Resolve type chains** - Follow offset references (e.g., const char* → pointer → const → base)
5. **Map to Julia types** - Base types mapped, complex types in progress
6. **Generate compilation_metadata.json** - Complete manifest for wrapper generation

**Verified Working (struct_test example):**
-  Struct layout extraction: `Point { double x; double y; }` (standard-layout)
-  Function signatures: `create_point(double, double) -> Point`
-  Return type extraction via DWARF DIEs
-  Generated Julia struct with correct field types
-  Direct ccall with struct-by-value (x86_64 System V ABI)

**Assumptions:**
- x86_64 Linux (System V ABI)
- Clang/GCC struct layout rules
- No padding removal under LTO
- DWARF accurately reflects compiled layout

**Test Command:**
```julia
cd("examples/struct_test")
include("julia/StructTest.jl")
using .StructTest
p1 = create_point(3.0, 4.0)
d = distance(p1, create_point(0.0, 0.0))  # Returns 25.0 (distance squared)
```

### Wrapper Generation Tiers

RepliBuild has a 3-tier wrapping system ([Wrapper.jl](src/Wrapper.jl)):

1. **Basic** (`:basic`): Symbol extraction only, minimal type info
2. **Advanced** (`:advanced`): Header-aware using Clang.jl
3. **Introspective** (`:introspective`): Full DWARF metadata with perfect types

Always prefer `:introspective` tier when metadata is available.

### Compilation Metadata

After compilation, RepliBuild generates `compilation_metadata.json` in the output directory.

**Real Example** (from examples/struct_test/julia/compilation_metadata.json):
```json
{
  "symbols": [
    {"mangled": "_Z12create_pointdd", "demangled": "create_point(double, double)", "type": "T"}
  ],
  "functions": [
    {
      "name": "create_point",
      "mangled": "_Z12create_pointdd",
      "demangled": "create_point(double, double)",
      "return_type": {"c_type": "Point", "julia_type": "Point"},
      "return_type_source": "dwarf",
      "parameters": [
        {"c_type": "double", "julia_type": "Cdouble", "position": 0},
        {"c_type": "double", "julia_type": "Cdouble", "position": 1}
      ]
    }
  ],
  "struct_definitions": {
    "Point": {
      "kind": "struct",
      "members": [
        {"name": "x", "c_type": "double", "julia_type": "Cdouble"},
        {"name": "y", "c_type": "double", "julia_type": "Cdouble"}
      ]
    }
  },
  "compiler_info": {
    "llvm_version": "21.1.5",
    "clang_version": "clang version 21.1.5",
    "optimization_level": "2",
    "target_triple": "x86_64-unknown-linux-gnu"
  }
}
```

This metadata drives the wrapper generator to create ABI-correct Julia bindings.

## Important Implementation Details

### Type Mapping (Compiler.jl:482-597)
The `dwarf_type_to_julia()` function maps C/C++ types to Julia types:
- Primitives: `int → Cint`, `double → Cdouble`, `bool → Bool`
- Pointers: `char* → Cstring`, `T* → Ptr{Cvoid}`
- Special handling for const/volatile qualifiers
- Unknown types default to `Any` for safety

### Safety Wrappers (Planned Feature)

**Current State:** Basic wrappers generate direct ccall statements with correct types.

**Planned Enhancements:**
- NULL pointer protection for Cstring returns
- Integer overflow protection with `Integer` parameters
- Ergonomic APIs with automatic conversions

**Current Generated Code** (examples/struct_test/julia/StructTest.jl):
```julia
function create_point(arg1::Cdouble, arg2::Cdouble)::Point
    ccall((:_Z12create_pointdd, LIBRARY_PATH), Point, (Cdouble, Cdouble,), arg1, arg2)
end
```

This is direct, zero-overhead, ABI-correct FFI. Safety features can be added as opt-in wrappers.

### Configuration Files
Projects use `replibuild.toml` for configuration:
```toml
[project]
name = "mylib"
root = "."

[compile]
source_files = ["main.cpp"]
flags = ["-std=c++17", "-fPIC", "-O2", "-g"]  # -g is CRITICAL for DWARF

[binary]
type = "shared"  # or "executable"

[paths]
output = "julia"
```

### Module Loading Order
The module load order in [RepliBuild.jl:14-32](src/RepliBuild.jl#L14-L32) is **critical**:
1. Infrastructure first (Paths, LLVM, Config, BuildBridge)
2. Then core modules (ASTWalker, Discovery, etc.)
3. ClangJLBridge before Compiler
4. Compiler and Wrapper before WorkspaceBuilder
5. REPL_API last

Do not reorder without understanding dependencies.

## Common Tasks

### Adding Support for New DWARF Tags
1. Edit `extract_dwarf_return_types()` in [Compiler.jl](src/Compiler.jl)
2. Add parsing logic for the new `DW_TAG_*` type
3. Update `resolve_type_chain()` if needed for type resolution
4. Add mapping in `dwarf_type_to_julia()` if new Julia type needed
5. Test with a C++ example that uses the new type

### Extending the Type Registry
1. Edit `dwarf_type_to_julia()` in [Compiler.jl:482-597](src/Compiler.jl#L482-L597)
2. Add new type mappings to the `type_map` dictionary
3. Update `get_type_size()` if size is needed
4. Test with both basic and struct types

### Adding New Wrapper Tier
1. Edit [Wrapper.jl](src/Wrapper.jl)
2. Add new tier to `WrapperTier` enum
3. Implement generation function following existing tier patterns
4. Update `wrap_library()` to handle new tier

### Debugging DWARF Extraction
```bash
# Manually inspect DWARF info:
readelf --debug-dump=info julia/libmylib.so | less

# Search for specific types:
readelf --debug-dump=info julia/libmylib.so | grep -A 5 "DW_TAG_structure_type"

# Check what symbols are exported:
nm -gC --defined-only julia/libmylib.so
```

## Testing Strategy

### Validated Examples

**1. struct_test** ([examples/struct_test/](examples/struct_test/))
- **Status:**  WORKING
- **Tests:** Struct-by-value parameters, struct returns, basic arithmetic
- **Verified:** Run this to confirm the system works:
```bash
cd examples/struct_test
julia -e 'include("julia/StructTest.jl"); using .StructTest;
          p1 = create_point(3.0, 4.0);
          println("Created: x=$(p1.x), y=$(p1.y)");
          d = distance(p1, create_point(0.0, 0.0));
          println("Distance²: $d")  # Outputs: Distance²: 25.0'
```

**2. Eigen Validation** (documented in [docs/EIGEN_VALIDATION.md](docs/EIGEN_VALIDATION.md))
- **Status:**  Type extraction successful (20,000+ types)
- **Achievement:** Proves DWARF parsing scales to complex template-heavy codebases
- **Note:** Full bindings generation is next phase

### Testing New Features

**Full Pipeline Test:**
```julia
cd("examples/struct_test")

# 1. Build C++ with debug info
using RepliBuild
RepliBuild.build(".")

# 2. Generate wrappers from metadata
RepliBuild.wrap("julia/libstruct_test.so", tier=:introspective)

# 3. Test the bindings
include("julia/StructTest.jl")
using .StructTest

# 4. Verify correctness
p1 = create_point(3.0, 4.0)
@assert p1.x == 3.0 && p1.y == 4.0
d = distance(p1, create_point(0.0, 0.0))
@assert d == 25.0  # Should be distance squared
```

**Expected Output:**
- Compilation completes with metadata saved
- Wrapper generation creates StructTest.jl
- Tests pass with correct numerical results

## Project Context

### Recent Major Changes (cleanup/phase1-simplification branch)
- Removed ~2700 LOC of redundant code (LLVMake.jl, JuliaWrapItUp.jl, ErrorLearning.jl, daemon system)
- **Bridge_LLVM.jl removed** → Replaced by Compiler.jl
- Fixed broken module references (RepliBuild.jl, REPL_API.jl, WorkspaceBuilder.jl)
- Consolidated to 13 source files (11 modules + 2 utilities) with clear separation of concerns
- Module loading order documented and critical

### Current Capabilities (Verified)
-  DWARF extraction: base types, pointers, const, references, standard-layout structs
-  Struct layout extraction: members, types (for standard-layout types)
-  Function signature extraction: parameters and return types
-  Struct-by-value passing: works for trivially-copyable types
-  DWARF parsing scales: 20K+ DIEs from Eigen extracted

### Explicit Limitations (See LIMITATIONS.md)
- ❌ Virtual methods, vtables, inheritance
- ❌ STL containers (ABI-unstable)
- ❌ Non-standard-layout types
- ❌ Exception specifications
- ❌ Types optimized out of DWARF

### Active Development
- 🔄 Enum support (DW_TAG_enumeration_type extractable)
- 🔄 Fixed-size array support (wrap as NTuple)
- 🔄 Validation tools (DWARF-IR cross-validation)

### Future Work (High Risk)
- STL containers: Experimental, vendor-specific, ABI-unstable
- Not recommended for production use

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical architecture and [docs/EIGEN_VALIDATION.md](docs/EIGEN_VALIDATION.md) for real validation data.

## Key Files to Read First

When understanding the codebase:
1. [README.md](README.md) - High-level overview and examples
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed technical architecture
3. [src/RepliBuild.jl](src/RepliBuild.jl) - Main module and public API
4. [src/Compiler.jl](src/Compiler.jl) - DWARF extraction (the core innovation)
5. [src/Wrapper.jl](src/Wrapper.jl) - Julia binding generation

## Style Conventions

- Use explicit `import` statements (not `using`) for parent module imports
- Functions that execute external commands go in BuildBridge
- Configuration loading goes through ConfigurationManager
- All user-facing functions should have docstrings
- Keep modules focused: one clear responsibility per module
