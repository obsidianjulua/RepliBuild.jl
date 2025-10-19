# RepliBuild.jl TOML Configuration Keys Analysis

## Overview
This document comprehensively traces all TOML configuration values used in RepliBuild.jl, organized by section with types, defaults, and usage patterns.

---

## [PROJECT] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `name` | String | `basename(project_root)` | Project name | ConfigurationManager.jl:127 |
| `root` | String | `config_dir` | Project root directory | ConfigurationManager.jl:128 |
| `uuid` | String | `uuid4()` (generated) | Unique project identifier | ConfigurationManager.jl:117-121 |

### Details
- **name**: Auto-generated from directory name if not specified
- **root**: Used as base for relative path resolution; defaults to directory containing replibuild.toml
- **uuid**: Generated once per project for tracking; preserved in TOML for consistency

---

## [DISCOVERY] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `true` | Enable discovery pipeline | ConfigurationManager.jl:277 |
| `scan_recursive` | Bool | `true` | Recursively scan directories | ConfigurationManager.jl:278 |
| `max_depth` | Int | `10` | Maximum directory recursion depth | ConfigurationManager.jl:279 |
| `exclude_dirs` | Vector{String} | `["build", ".git", ".cache", "node_modules"]` | Directories to skip during scanning | ConfigurationManager.jl:280 |
| `follow_symlinks` | Bool | `false` | Follow symbolic links | ConfigurationManager.jl:281 |
| `parse_ast` | Bool | `true` | Parse AST during discovery | ConfigurationManager.jl:282 |
| `walk_dependencies` | Bool | `true` | Walk dependency graph | ConfigurationManager.jl:283 |
| `log_all_files` | Bool | `true` | Log every discovered file | ConfigurationManager.jl:284 |
| `completed` | Bool | `true` (after discovery) | Whether discovery has completed | Discovery.jl:459 |
| `timestamp` | String | `string(now())` | When discovery last ran | Discovery.jl:460 |
| `files` | Dict{String,Vector{String}} | Empty dict | Categorized source files discovered | Discovery.jl:461-467 |
| `include_dirs` | Vector{String} | Empty array | Discovered include directories | Discovery.jl:474, ConfigurationManager.jl:500 |
| `binaries` | Dict | Empty dict | Detected binaries info | Discovery.jl:468-473 |
| `dependency_graph_file` | String | `""` or `.replibuild_cache/dependency_graph.json` | Path to saved dependency graph | Discovery.jl:475 |

### Sub-keys under `files`
- **cpp_sources**: Vector{String} - C++ source files (.cpp, .cc, .cxx, .c++)
- **cpp_headers**: Vector{String} - C++ header files (.hpp, .hxx, .h++, .hh)
- **c_sources**: Vector{String} - C source files (.c)
- **c_headers**: Vector{String} - C header files (.h)
- **total_scanned**: Int - Total files scanned

### Sub-keys under `binaries`
- **executables**: Vector{String} - Found executable files
- **static_libs**: Vector{String} - Found .a libraries
- **shared_libs**: Vector{String} - Found .so libraries
- **total**: Int - Total binaries found

### Details
- `files` contains vectorized results from file scanning
- `include_dirs` stored in discovery, accessible via `ConfigurationManager.get_include_dirs()`
- `dependency_graph` stored as separate JSON, reference stored in TOML

---

## [REORGANIZE] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `false` | Enable reorganization stage | ConfigurationManager.jl:288 |
| `create_structure` | Bool | `true` | Create new directory structure | ConfigurationManager.jl:289 |
| `sort_by_type` | Bool | `true` | Sort files by type | ConfigurationManager.jl:290 |
| `preserve_hierarchy` | Bool | `false` | Preserve original directory hierarchy | ConfigurationManager.jl:291 |
| `target_structure` | Dict | See below | Target directory layout | ConfigurationManager.jl:292-300 |

### Sub-keys under `target_structure`
- **cpp_sources**: String (default: `"src"`) - Target for C++ sources
- **cpp_headers**: String (default: `"include"`) - Target for C++ headers
- **c_sources**: String (default: `"src"`) - Target for C sources
- **c_headers**: String (default: `"include"`) - Target for C headers
- **julia_files**: String (default: `"julia"`) - Target for Julia files
- **config_files**: String (default: `"config"`) - Target for config files
- **docs**: String (default: `"docs"`) - Target for documentation

### Details
- Reorganize is optional and disabled by default
- Only applied if explicitly enabled

---

## [COMPILE] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `true` | Enable compilation stage | ConfigurationManager.jl:304 |
| `output_dir` | String | `"build/ir"` | Output directory for IR files | ConfigurationManager.jl:305 |
| `flags` | Vector{String} | `["-std=c++17", "-fPIC"]` | Compiler flags | ConfigurationManager.jl:306, LLVMake.jl:87 |
| `include_dirs` | Vector{String} | Empty array | Include directories for compilation | ConfigurationManager.jl:307, LLVMake.jl:163 |
| `defines` | Dict{String,String} | Empty dict | Preprocessor defines | ConfigurationManager.jl:308, LLVMake.jl:166 |
| `emit_ir` | Bool | `true` | Generate LLVM IR files | ConfigurationManager.jl:309 |
| `emit_bc` | Bool | `false` | Generate LLVM bitcode files | ConfigurationManager.jl:310 |
| `parallel` | Bool | `true` | Enable parallel compilation | ConfigurationManager.jl:311 |
| `source_files` | Vector{String} | Empty array | Specific source files to compile | Discovery.jl:144 |
| `lib_dirs` | Vector{String} | Empty array | Library search directories | LLVMake.jl:164 |
| `libraries` | Vector{String} | Empty array | External libraries to link | LLVMake.jl:165 |
| `extra_flags` | Vector{String} | Empty array | Additional compiler flags | LLVMake.jl:167 |
| `walk_dependencies` | Bool | `true` | Walk header dependencies | Bridge_LLVM.jl:89 |
| `max_depth` | Int | `10` | Max dependency walk depth | Bridge_LLVM.jl:90 |
| `link_libraries` | Vector{String} | Empty array | Libraries to link against | Bridge_LLVM.jl (line ~650) |

### Details
- **flags**: Validated to be string vector (ConfigurationManager.jl:712-722)
- **include_dirs**: Paths validated to exist (ConfigurationManager.jl:726-735)
- **defines**: String→String dictionary for C preprocessor defines
- All paths can be relative (resolved from project_root) or absolute

---

## [LINK] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `true` | Enable linking stage | ConfigurationManager.jl:315 |
| `output_dir` | String | `"build/linked"` | Output directory for linked IR | ConfigurationManager.jl:316 |
| `optimize` | Bool | `true` | Enable optimization | ConfigurationManager.jl:317 |
| `opt_level` | String | `"O2"` | Optimization level (O0, O1, O2, O3, Os, Oz) | ConfigurationManager.jl:318, LLVMake.jl:155 |
| `opt_passes` | Vector{String} | Empty array | Custom LLVM optimization passes | ConfigurationManager.jl:319 |
| `lto` | Bool | `false` | Enable link-time optimization | ConfigurationManager.jl:320, LLVMake.jl:157 |

### Details
- **opt_level**: Affects optimization aggressiveness during linking
- **opt_passes**: Allows specifying custom LLVM pass manager passes

---

## [BINARY] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `true` | Enable binary creation stage | ConfigurationManager.jl:324 |
| `output_dir` | String | `"julia"` | Output directory for shared library | ConfigurationManager.jl:325 |
| `library_name` | String | `"lib" + lowercase(project_name) + ".so"` | Output library filename | ConfigurationManager.jl:326, Discovery.jl:501 |
| `library_type` | String | `"shared"` | Library type: "shared" or "static" | ConfigurationManager.jl:327 |
| `link_libraries` | Vector{String} | Empty array | Additional libraries to link | ConfigurationManager.jl:328 |
| `rpath` | Bool | `true` | Enable RPATH for library location | ConfigurationManager.jl:329 |

### Details
- **library_name**: Auto-generated if empty; formatted as `lib{name}.so`
- **library_type**: Currently supports "shared" (default) and "static"
- **rpath**: Enables finding dependencies relative to library location

---

## [SYMBOLS] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `true` | Enable symbol extraction stage | ConfigurationManager.jl:333 |
| `method` | String | `"nm"` | Symbol extraction method: "nm", "objdump", "llvm-nm" | ConfigurationManager.jl:334 |
| `demangle` | Bool | `true` | Demangle C++ symbol names | ConfigurationManager.jl:335 |
| `filter_internal` | Bool | `true` | Filter internal/hidden symbols | ConfigurationManager.jl:336 |
| `export_list` | Bool | `true` | Export symbol list | ConfigurationManager.jl:337 |

### Details
- **method**: Selects which tool to use for extracting symbols
- **demangle**: Converts mangled C++ names to readable form
- **filter_internal**: Removes symbols like `_ZN...` internal markers

---

## [WRAP] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `true` | Enable Julia wrapper generation | ConfigurationManager.jl:341 |
| `output_dir` | String | `"julia"` | Output directory for wrappers | ConfigurationManager.jl:342 |
| `style` | String | `"auto"` | Wrapper style: "auto", "basic", "advanced", "clangjl" | ConfigurationManager.jl:343 |
| `module_name` | String | `uppercasefirst(project_name)` | Generated Julia module name | ConfigurationManager.jl:344, Discovery.jl:517 |
| `add_tests` | Bool | `true` | Generate test file | ConfigurationManager.jl:345 |
| `add_docs` | Bool | `true` | Generate documentation | ConfigurationManager.jl:346 |
| `type_mappings` | Dict{String,String} | Empty dict | C++ → Julia type mappings | ConfigurationManager.jl:347 |

### Details
- **style**: Determines wrapper generation strategy
  - "auto": Automatically select best method
  - "basic": Simple ccall-based wrappers
  - "advanced": More sophisticated wrappers with type conversions
  - "clangjl": Use Clang.jl integration
- **module_name**: Auto-generated from project name if empty
- **type_mappings**: Maps C++ types to Julia types for generation

---

## [TEST] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `false` | Enable testing stage | ConfigurationManager.jl:351 |
| `test_dir` | String | `"test"` | Directory containing tests | ConfigurationManager.jl:352 |
| `run_tests` | Bool | `false` | Automatically run tests | ConfigurationManager.jl:353 |

### Details
- Test stage is disabled by default
- **test_dir**: Location of test files relative to project root
- **run_tests**: If true, executes tests after build completes

---

## [LLVM] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `use_replibuild_llvm` | Bool | `true` | Use bundled RepliBuild LLVM | ConfigurationManager.jl:357 |
| `isolated` | Bool | `true` | Use isolated LLVM environment | ConfigurationManager.jl:358 |
| `root` | String | `""` (auto-detected) | Path to LLVM installation | LLVMEnvironment.jl:134-142, 432-445 |
| `source` | String | `"intree"` | LLVM source: "intree", "jll", "custom", "system" | LLVMEnvironment.jl:137, 435, 435 |
| `tools` | Dict{String,String} | Auto-discovered | Tool name → path mappings | LLVMEnvironment.jl:482-494, 485 |

### Details
- **root**: Priority order for auto-detection:
  1. User TOML override (if valid)
  2. LLVM_full_assert_jll (if installed)
  3. In-tree RepliBuild LLVM
  4. System LLVM (/usr, /usr/local, /opt/llvm, etc.)
- **source**: Indicates which LLVM distribution is used
  - "intree": Local RepliBuild LLVM in LLVM/ subdirectory
  - "jll": Julia package LLVM_full_assert_jll
  - "custom": User-specified path
  - "system": Found in system directories
- **tools**: Dictionary mapping tool names to absolute paths
  - Keys: "clang", "clang++", "llvm-config", "llvm-link", "opt", "llc", "nm", "objdump", etc.
  - Populated by `discover_llvm_tools()` during initialization

---

## [TARGET] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `triple` | String | `""` (host triple) | Target triple (e.g., "x86_64-unknown-linux-gnu") | LLVMake.jl:152, Bridge_LLVM.jl:91 |
| `cpu` | String | `"generic"` | Target CPU type | LLVMake.jl:153, Bridge_LLVM.jl:92 |
| `features` | Vector{String} | Empty array | CPU features (+avx2, +fma, -sse4.2, etc.) | LLVMake.jl:154, Bridge_LLVM.jl (line needs checking) |
| `opt_level` | String | `"O2"` | Optimization level | Bridge_LLVM.jl:93 |
| `debug` | Bool | `false` | Include debug symbols | LLVMake.jl:156 |
| `lto` | Bool | `false` | Link-time optimization | LLVMake.jl:157, Bridge_LLVM.jl:94 |
| `sanitizers` | Vector{String} | Empty array | Enabled sanitizers: "address", "thread", "memory", "undefined" | LLVMake.jl:158 |

### Details
- **triple**: Empty defaults to host machine triple
- **cpu**: Examples: "generic", "native", "haswell", "skylake", etc.
- **features**: CPU-specific features for more precise targeting
- **sanitizers**: For debugging, adds instrumentation for memory/thread errors

---

## [WORKFLOW] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `stages` | Vector{String} | `["discovery", "compile", "link", "binary", "symbols", "wrap"]` | Stages to execute in order | ConfigurationManager.jl:368, Bridge_LLVM.jl:95 |
| `stop_on_error` | Bool | `true` | Stop build if stage fails | ConfigurationManager.jl:369 |
| `parallel_stages` | Vector{String} | `["compile"]` | Stages that can run in parallel | ConfigurationManager.jl:370 |
| `parallel` | Bool | `true` | Enable parallel execution | Bridge_LLVM.jl:96 |

### Details
- **stages**: Controls which build stages execute and in what order
- **stop_on_error**: If false, continues to next stage on failure
- **parallel_stages**: Specifies stages that support parallel execution

---

## [CACHE] SECTION

### Keys and Usage

| Key | Type | Default | Purpose | Usage Location |
|-----|------|---------|---------|-----------------|
| `enabled` | Bool | `true` | Enable build caching | ConfigurationManager.jl:374 |
| `directory` | String | `".replibuild_cache"` | Cache directory location | ConfigurationManager.jl:375 |
| `invalidate_on_change` | Bool | `true` | Invalidate cache if inputs change | ConfigurationManager.jl:376 |

### Details
- **directory**: Relative path from project_root where cache is stored
- **invalidate_on_change**: If true, automatically detects file modifications

---

## Configuration Access Patterns

### Primary Access Method
```julia
get(config.section, "key", default_value)
```

### Common Type Conversions

#### Vector Handling
```julia
# String arrays from TOML
flags = get(config.compile, "flags", String[])

# Converting TOML empty arrays
features = String[get(config.target, "features", String[])...]
```

#### Dict Handling
```julia
# Nested dictionaries
defines = Dict(String(k) => String(v) for (k, v) in get(config.compile, "defines", Dict()))
```

#### Validation Patterns
```julia
# Check if key exists and is non-empty
if haskey(config.discovery, "include_dirs") && !isempty(config.discovery["include_dirs"])
    # Use the value
end

# Check specific key existence
if haskey(config.llvm, "tools") && !isempty(config.llvm["tools"])
    # Use cached tools
else
    # Auto-discover tools
end
```

---

## Validation Rules (from ConfigurationManager.jl)

1. **project.name**: Cannot be empty
2. **project.root**: Must be valid directory
3. **output_dir** (for compile, link, binary, wrap): Must be within project or relative
4. **compile.flags**: Must be vector of strings
5. **compile.include_dirs**: Directories must exist
6. **compile.sources**: Source files must exist
7. **Relative paths**: Resolved from project.root

---

## File Organization

- **ConfigurationManager.jl**: Defines structure, defaults, and validation
- **Discovery.jl**: Populates discovery section during initial scan
- **LLVMEnvironment.jl**: Manages LLVM section
- **LLVMake.jl**: Reads compile, target, bindings sections
- **Bridge_LLVM.jl**: Reads from multiple sections for orchestration
- **BuildBridge.jl**: Uses configuration for build execution

---

## Default Configuration Template

```toml
version = "0.1.0"

[project]
name = "project_name"
root = "."
uuid = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

[discovery]
enabled = true
scan_recursive = true
max_depth = 10
exclude_dirs = ["build", ".git", ".cache", "node_modules"]
follow_symlinks = false
parse_ast = true
walk_dependencies = true
log_all_files = true

[reorganize]
enabled = false
create_structure = true
sort_by_type = true
preserve_hierarchy = false

[compile]
enabled = true
output_dir = "build/ir"
flags = ["-std=c++17", "-fPIC"]
include_dirs = []
defines = {}
emit_ir = true
emit_bc = false
parallel = true

[link]
enabled = true
output_dir = "build/linked"
optimize = true
opt_level = "O2"
opt_passes = []
lto = false

[binary]
enabled = true
output_dir = "julia"
library_name = ""
library_type = "shared"
link_libraries = []
rpath = true

[symbols]
enabled = true
method = "nm"
demangle = true
filter_internal = true
export_list = true

[wrap]
enabled = true
output_dir = "julia"
style = "auto"
module_name = ""
add_tests = true
add_docs = true
type_mappings = {}

[test]
enabled = false
test_dir = "test"
run_tests = false

[llvm]
use_replibuild_llvm = true
isolated = true
root = ""
source = ""
tools = {}

[target]
triple = ""
cpu = "generic"
features = []
opt_level = "O2"
debug = false
lto = false
sanitizers = []

[workflow]
stages = ["discovery", "compile", "link", "binary", "symbols", "wrap"]
stop_on_error = true
parallel_stages = ["compile"]
parallel = true

[cache]
enabled = true
directory = ".replibuild_cache"
invalidate_on_change = true
```
