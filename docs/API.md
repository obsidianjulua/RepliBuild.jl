# RepliBuild API Reference

**Complete API documentation for RepliBuild v2.0.1**

This is the ONLY API reference you need. Everything here is tested and works.

---

## Table of Contents

1. [User API](#user-api) - 3 functions you actually need
2. [Module Structure](#module-structure) - What's inside
3. [Advanced API](#advanced-api) - For power users
4. [Architecture](#architecture) - How it works

---

## User API

### build(path="."; clean=false) → String

Compile C++ project to shared library.

**What it does:**
1. Compiles C++ → LLVM IR (parallel if enabled)
2. Links and optimizes IR
3. Generates library (.so/.dylib/.dll)
4. Extracts metadata (DWARF + symbols)

**Arguments:**
- `path::String` - Project directory (default: ".")
- `clean::Bool` - Clean before building (default: false)

**Returns:**
- Library path (String)

**Example:**
```julia
using RepliBuild

# Build library
lib_path = RepliBuild.build()
# => "julia/libmyproject.so"

# Clean build
RepliBuild.build(clean=true)
```

**Requirements:**
- `replibuild.toml` must exist in project root
- C++ source files in configured directories
- LLVM toolchain available

---

### wrap(path="."; headers=String[]) → String

Generate Julia wrapper from compiled library.

**What it does:**
1. Loads metadata from build
2. Generates Julia module with ccall wrappers
3. Creates type definitions
4. Saves to julia/ directory

**Arguments:**
- `path::String` - Project directory (default: ".")
- `headers::Vector{String}` - C++ headers for enhanced wrapping (optional)

**Returns:**
- Path to generated wrapper file (String)

**Example:**
```julia
using RepliBuild

# Generate wrapper (requires build() first)
wrapper = RepliBuild.wrap()
# => "julia/MyProject.jl"

# Use the wrapper
include(wrapper)
using .MyProject

result = my_cpp_function(42)
```

**Requirements:**
- Must run `build()` first
- Library must exist in output directory
- Metadata file recommended (but optional)

---

### info(path=".") → Nothing

Show project status and build artifacts.

**What it shows:**
- Project name from config
- Library status (built/not built)
- Wrapper status (generated/not generated)

**Example:**
```julia
using RepliBuild

RepliBuild.info()
# ══════════════════════════════════════════════════════════════════════
#  RepliBuild - Project Info
# ══════════════════════════════════════════════════════════════════════
# Project: MyProject
#
# ✓ Library: libmyproject.so
# ✓ Wrapper: MyProject.jl
# ══════════════════════════════════════════════════════════════════════
```

---

### clean(path=".") → Nothing

Remove all build artifacts.

**What it removes:**
- `build/` - IR files, object files
- `julia/` - Generated library and wrappers
- `.replibuild_cache/` - Cached data

**Example:**
```julia
RepliBuild.clean()
# ✓ Removed build/
# ✓ Removed julia/
# ✓ Removed .replibuild_cache/
```

---

## Module Structure

RepliBuild consists of 11 internal modules organized by responsibility:

### User-Facing Modules

#### Compiler

Orchestrates C++ → LLVM IR → Library compilation.

**Exports:**
- `compile_project(config)` - Full compilation workflow
- `compile_to_ir(config, files)` - C++ → IR
- `link_optimize_ir(config, ir_files, name)` - IR → optimized IR
- `create_library(config, ir_file)` - IR → .so
- `create_executable(config, ir_file, name)` - IR → executable

**Submodules:**
- `IRCompiler` - Compilation and linking
- `DWARFExtractor` - Debug info parsing
- `MetadataExtractor` - Symbol extraction

**Access:**
```julia
using RepliBuild.Compiler

# Compile specific files to IR
config = ConfigurationManager.load_config("replibuild.toml")
ir_files = Compiler.compile_to_ir(config, ["src/foo.cpp", "src/bar.cpp"])

# Link and optimize
linked_ir = Compiler.link_optimize_ir(config, ir_files, "mylib")

# Create library
lib_path = Compiler.create_library(config, linked_ir)
```

---

#### Wrapper

Generates Julia bindings from library + metadata.

**Exports:**
- `wrap_library(config, lib_path; headers, tier, generate_tests, generate_docs)`
- `generate_module_from_metadata(metadata, lib_path, module_name)`

**Example:**
```julia
using RepliBuild.Wrapper

config = ConfigurationManager.load_config("replibuild.toml")
wrapper_path = Wrapper.wrap_library(
    config,
    "julia/libmylib.so",
    headers=["include/mylib.h"],
    generate_docs=true
)
```

---

#### Discovery

Automatic project detection and configuration generation.

**Exports:**
- `discover(path; force=false)` - Auto-detect C++ project structure
- `import_cmake(path)` - Import from CMakeLists.txt

**Example:**
```julia
using RepliBuild.Discovery

# Auto-detect project
Discovery.discover(".")
# => Creates replibuild.toml

# Import from CMake
Discovery.import_cmake(".")
```

---

#### ConfigurationManager

Configuration file loading and validation.

**Exports:**
- `load_config(path)` - Load replibuild.toml → RepliBuildConfig
- `RepliBuildConfig` - Configuration struct
- Getters: `get_source_files`, `get_include_dirs`, `get_compile_flags`, etc.

**Example:**
```julia
using RepliBuild.ConfigurationManager

config = load_config("replibuild.toml")
sources = get_source_files(config)
includes = get_include_dirs(config)
flags = get_compile_flags(config)
```

---

### Internal Modules

These are used internally and rarely needed by users:

- **RepliBuildPaths** - Path management and cache directories
- **LLVMEnvironment** - LLVM toolchain detection and setup
- **BuildBridge** - Subprocess execution (clang++, llvm-link, etc.)
- **ASTWalker** - C++ AST traversal (not currently used)
- **CMakeParser** - CMakeLists.txt parsing
- **ClangJLBridge** - Clang integration utilities
- **WorkspaceBuilder** - Multi-library workspace support (experimental)

---

## Advanced API

### Compiler Module Functions

#### compile_to_ir(config, cpp_files) → Vector{String}

Compile C++ files to LLVM IR.

**Features:**
- Parallel compilation (if enabled in config)
- Incremental builds (skips unchanged files)
- Caching

**Returns:** Vector of IR file paths

---

#### link_optimize_ir(config, ir_files, output_name) → String

Link IR files and optionally optimize.

**Features:**
- Uses llvm-link
- Optional optimization (opt)
- LTO support

**Returns:** Path to linked IR file

---

#### create_library(config, ir_file, lib_name="") → String

Create shared library from IR.

**Returns:** Path to .so/.dylib/.dll file

---

#### create_executable(config, ir_file, exe_name, link_libraries=[], link_flags=[]) → String

Create executable from IR.

**Returns:** Path to executable

---

### DWARFExtractor Module Functions

#### extract_dwarf_return_types(binary_path) → (Dict, Dict)

Extract type information from DWARF debug info.

**Returns:**
- return_types: Dict{mangled_name => type_info}
- enums: Dict{enum_name => enum_info}

**Type Info Structure:**
```julia
{
  "c_type" => "double",
  "julia_type" => "Cdouble",
  "size" => 8,
  "parameters" => [
    {"name" => "x", "c_type" => "int", "julia_type" => "Cint"}
  ]
}
```

---

#### dwarf_type_to_julia(c_type) → String

Convert C/C++ type to Julia FFI type.

**Examples:**
```julia
dwarf_type_to_julia("int")          # => "Cint"
dwarf_type_to_julia("double*")      # => "Ptr{Cvoid}"
dwarf_type_to_julia("char*")        # => "Cstring"
dwarf_type_to_julia("const int&")   # => "Ref{Cvoid}"
```

---

#### get_type_size(c_type) → Int

Get type size in bytes.

**Example:**
```julia
get_type_size("int")      # => 4
get_type_size("double")   # => 8
get_type_size("int*")     # => 8 (on x86_64)
```

---

### MetadataExtractor Module Functions

#### extract_compilation_metadata(config, source_files, binary_path) → Dict

Extract complete metadata from binary.

**What it extracts:**
- Symbols (mangled + demangled)
- Function signatures
- Return types (from DWARF)
- Enums
- Type registry

**Returns:** Metadata dict with structure:
```julia
{
  "project" => "MyProject",
  "module_name" => "MyProject",
  "generated_at" => "2025-11-26 12:00:00",
  "source_files" => [...],
  "binary_path" => "julia/libmyproject.so",
  "symbols" => [...],
  "functions" => [...],
  "enums" => {...},
  "type_registry" => {...},
  "function_count" => 42,
  "symbol_count" => 56
}
```

---

#### save_compilation_metadata(config, source_files, binary_path) → String

Extract and save metadata to JSON.

**Returns:** Path to compilation_metadata.json

---

### ConfigurationManager Functions

#### get_source_files(config) → Vector{String}

Get all C++ source files from config.

---

#### get_include_dirs(config) → Vector{String}

Get include directories.

---

#### get_compile_flags(config) → Vector{String}

Get compiler flags (optimization, warnings, etc.).

---

#### get_build_path(config) → String

Get build directory path.

---

#### get_output_path(config) → String

Get output directory path (where library goes).

---

#### get_library_name(config) → String

Get library filename with platform extension.

---

#### get_module_name(config) → String

Get Julia module name.

---

#### is_parallel_enabled(config) → Bool

Check if parallel compilation is enabled.

---

#### is_cache_enabled(config) → Bool

Check if incremental build caching is enabled.

---

## Architecture

### Compilation Pipeline

```
┌─────────────┐
│   build()   │  User calls this
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Compiler.compile_project(config)    │
└──────┬───────────────────────────────┘
       │
       ├─► 1. IRCompiler.compile_to_ir(config, cpp_files)
       │      ├─ Parallel if enabled
       │      ├─ Incremental caching
       │      └─► C++ files → .ll IR files
       │
       ├─► 2. IRCompiler.link_optimize_ir(config, ir_files, name)
       │      ├─ llvm-link
       │      ├─ opt (if enabled)
       │      └─► IR files → single optimized .ll
       │
       ├─► 3. IRCompiler.create_library(config, ir_file)
       │      └─► IR → libname.so
       │
       └─► 4. MetadataExtractor.save_compilation_metadata(...)
              ├─ nm: extract symbols
              ├─ readelf: parse DWARF
              ├─ DWARFExtractor: type info
              └─► compilation_metadata.json
```

### Wrapper Generation Pipeline

```
┌─────────────┐
│    wrap()   │  User calls this
└──────┬──────┘
       │
       ▼
┌────────────────────────────────────┐
│  Wrapper.wrap_library(...)         │
└──────┬─────────────────────────────┘
       │
       ├─► 1. Load compilation_metadata.json
       │      ├─ Functions with signatures
       │      ├─ Return types (DWARF)
       │      ├─ Enums
       │      └─ Type registry
       │
       ├─► 2. Generate Julia code
       │      ├─ Module skeleton
       │      ├─ ccall wrappers for each function
       │      ├─ Enum definitions
       │      ├─ Type definitions
       │      └─ Documentation comments
       │
       └─► 3. Save to julia/ModuleName.jl
```

### Module Dependencies

```
RepliBuild (main)
├── ConfigurationManager ─────┐
│   └── Uses: RepliBuildPaths │
│                              │
├── Compiler ─────────────────┤
│   ├── IRCompiler            │
│   │   └── Uses: BuildBridge ├──► BuildBridge (subprocess execution)
│   ├── DWARFExtractor         │
│   │   └── Uses: BuildBridge ─┘
│   └── MetadataExtractor
│       └── Uses: DWARFExtractor
│
├── Wrapper
│   └── Uses: ConfigurationManager
│
└── Discovery
    ├── Uses: ConfigurationManager
    └── Uses: CMakeParser
```

### Configuration File Structure

**replibuild.toml:**
```toml
[project]
name = "MyProject"
version = "1.0.0"

[paths]
source_dirs = ["src"]
include_dirs = ["include"]
build_dir = "build"
output_dir = "julia"

[compile]
std = "c++17"
optimization = "-O2"
warnings = ["-Wall", "-Wextra"]
defines = {}

[link]
enable_optimization = true
optimization_level = "-O2"
enable_lto = false
link_libraries = []

[binary]
type = "library"  # or "executable"

[parallel]
enabled = true
max_workers = 0  # 0 = auto (num cores)

[cache]
enabled = true
```

---

## Type Mapping Reference

### Primitive Types

| C/C++ Type | Julia Type | Size |
|------------|------------|------|
| `void` | `Cvoid` | 0 |
| `bool` | `Bool` | 1 |
| `char` | `Cchar` | 1 |
| `unsigned char` | `Cuchar` | 1 |
| `short` | `Cshort` | 2 |
| `unsigned short` | `Cushort` | 2 |
| `int` | `Cint` | 4 |
| `unsigned int` | `Cuint` | 4 |
| `long` | `Clong` | 8 |
| `unsigned long` | `Culong` | 8 |
| `long long` | `Clonglong` | 8 |
| `unsigned long long` | `Culonglong` | 8 |
| `float` | `Cfloat` | 4 |
| `double` | `Cdouble` | 8 |
| `size_t` | `Csize_t` | 8 |

### Pointer Types

| C/C++ Type | Julia Type |
|------------|------------|
| `T*` | `Ptr{Cvoid}` |
| `char*` | `Cstring` |
| `const char*` | `Cstring` |
| `T&` | `Ref{Cvoid}` |

### Fixed-Size Integer Types

| C/C++ Type | Julia Type |
|------------|------------|
| `int8_t` | `Int8` |
| `uint8_t` | `UInt8` |
| `int16_t` | `Int16` |
| `uint16_t` | `UInt16` |
| `int32_t` | `Int32` |
| `uint32_t` | `UInt32` |
| `int64_t` | `Int64` |
| `uint64_t` | `UInt64` |

---

## Common Workflows

### 1. New Project Setup

```julia
using RepliBuild

# Auto-detect C++ project
cd("my-cpp-project")
RepliBuild.Discovery.discover(".")

# Edit replibuild.toml if needed
# vim replibuild.toml

# Build library
RepliBuild.build()

# Generate wrapper
RepliBuild.wrap()

# Use it
include("julia/MyProject.jl")
using .MyProject
```

### 2. Iterative Development

```julia
using RepliBuild

# Edit C++ code...

# Rebuild (incremental)
RepliBuild.build()  # Only recompiles changed files

# Regenerate wrapper
RepliBuild.wrap()

# Test changes
include("julia/MyProject.jl")
using .MyProject
```

### 3. Clean Build

```julia
using RepliBuild

# Full clean rebuild
RepliBuild.clean()
RepliBuild.build()
RepliBuild.wrap()
```

### 4. Import from CMake

```julia
using RepliBuild

# Import existing CMake project
RepliBuild.Discovery.import_cmake(".")

# Review generated config
# vim replibuild.toml

# Build
RepliBuild.build()
RepliBuild.wrap()
```

---

## Error Handling

### Common Errors

**"Library not found" when calling wrap():**
```
Solution: Run RepliBuild.build() first
```

**"No replibuild.toml found":**
```
Solution: Run RepliBuild.Discovery.discover(".") first
```

**Compilation failures:**
```
Check:
- C++ syntax errors in source files
- Missing include directories in config
- Incompatible compiler flags
```

**"DWARF info not found" warning:**
```
Reason: Binary compiled without debug symbols
Solution: Add "-g" to compile.flags in replibuild.toml
Impact: Wrapper will have less accurate type information
```

---

## Performance Tips

### 1. Enable Parallel Compilation

```toml
[parallel]
enabled = true
max_workers = 0  # Auto-detect CPU cores
```

### 2. Enable Caching

```toml
[cache]
enabled = true
```

This enables incremental builds - only changed files recompile.

### 3. Disable Optimization for Debug Builds

```toml
[compile]
optimization = "-O0"

[link]
enable_optimization = false
```

Fast compilation, easier debugging, slower runtime.

### 4. Enable LTO for Release Builds

```toml
[link]
enable_optimization = true
optimization_level = "-O3"
enable_lto = true
```

Slower compilation, maximum runtime performance.

---

## Metadata Format

### compilation_metadata.json Structure

```json
{
  "project": "MyProject",
  "module_name": "MyProject",
  "generated_at": "2025-11-26 12:00:00",
  "source_files": ["src/foo.cpp", "src/bar.cpp"],
  "binary_path": "julia/libmyproject.so",

  "symbols": [
    {
      "mangled": "_Z3addii",
      "demangled": "add(int, int)",
      "type": "T"
    }
  ],

  "functions": [
    {
      "mangled_name": "_Z3addii",
      "demangled_name": "add(int, int)",
      "name": "add",
      "class": "",
      "return_type": {
        "c_type": "int",
        "julia_type": "Cint",
        "size": 4,
        "parameters": [
          {"name": "a", "c_type": "int", "julia_type": "Cint"},
          {"name": "b", "c_type": "int", "julia_type": "Cint"}
        ]
      }
    }
  ],

  "enums": {
    "Color": {
      "underlying_type": "unsigned int",
      "values": {
        "Red": 0,
        "Green": 1,
        "Blue": 2
      }
    }
  },

  "type_registry": {
    "base_types": {
      "int": "Cint",
      "double": "Cdouble"
    }
  },

  "function_count": 12,
  "symbol_count": 18
}
```

---

## Version History

- **v2.0.1** (Current)
  - Modularized Compiler.jl into IRCompiler, DWARFExtractor, MetadataExtractor
  - Improved DWARF parsing (enums, arrays, structs, function parameters)
  - Enhanced type validation and error handling
  - Simplified user API to 4 functions: build, wrap, info, clean

- **v1.0.0**
  - Initial release
  - Basic C++ → Julia FFI generation
  - LLVM IR compilation pipeline

---

## Credits

Built with:
- LLVM toolchain (clang++, llvm-link, opt)
- DWARF debug format (readelf)
- GNU binutils (nm)
- Julia ccall FFI

---

**End of API Reference**
