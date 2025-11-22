# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RepliBuild.jl is a C++ to Julia build orchestration system that provides dependency-aware parallel compilation for single and multi-library C++ projects. It discovers C++ project structure, compiles to LLVM IR, links libraries, and generates Julia bindings automatically.

**Key Philosophy**: Intelligent build orchestration through AST analysis and dependency graphs, supporting both single-library projects and CMake-style multi-library workspaces.

## Development Commands

### Running Tests
Currently no test suite exists (`test/runtests.jl` has been deleted). Tests need to be re-implemented.

### Building and Development
This is a Julia package that builds C++ projects. Development workflow:

```bash
# Enter Julia REPL
julia --threads=auto

# Load package in development mode
using Pkg; Pkg.activate("."); using RepliBuild

# Test the API
RepliBuild.discover("/path/to/cpp/project")
RepliBuild.build("/path/to/cpp/project")
```

### Running with REPL API
```julia
using RepliBuild

# Quick commands
rbuild()              # Build current project
rdiscover()           # Discover project structure
rclean()              # Clean build artifacts
rinfo()               # Show project info
rthreads()            # Check available threads
```

## Architecture

### Module Loading Order (Critical)

The main module `RepliBuild.jl` loads submodules in a specific order due to dependencies:

1. **RepliBuildPaths.jl** - Path management (no dependencies)
2. **LLVMEnvironment.jl** - LLVM toolchain discovery
3. **ConfigurationManager.jl** - TOML config handling
4. **BuildBridge.jl** - Tool discovery and execution wrapper
5. **ASTWalker.jl** - C++ dependency analysis via clang
6. **Discovery.jl** - Project scanning and analysis (uses ASTWalker, LLVMEnvironment)
7. **CMakeParser.jl** - CMake import functionality
8. **ClangJLBridge.jl** - Clang.jl integration for type-aware bindings
9. **Bridge_LLVM.jl** - Main compilation orchestrator (uses BuildBridge, ClangJLBridge)
10. **WorkspaceBuilder.jl** - Multi-library workspace builds (uses Bridge_LLVM)
11. **REPL_API.jl** - User-friendly REPL commands

**Important**: When adding new modules, respect this dependency order. Modules are loaded via `include()` and then imported with `using .ModuleName` in RepliBuild.jl.

### Core Pipeline: Discovery → Build → Wrap

**Discovery Phase** (`Discovery.jl`)
- Scans project for C++ sources and headers
- Uses `ASTWalker.jl` to build dependency graphs via clang AST
- Detects include directories and binaries
- Generates `replibuild.toml` configuration
- Caches results in `.replibuild_cache/`

**Build Phase** (`Bridge_LLVM.jl`, `WorkspaceBuilder.jl`)
- Detects single-library vs multi-library workspace structure
- For workspaces: uses topological sort for parallel build ordering (Kahn's algorithm)
- Compiles C++ → LLVM IR with incremental caching
- Links IR files and optimizes
- Creates shared libraries (.so) or executables
- Supports parallel compilation using Julia threads

**Wrapping Phase** (`ClangJLBridge.jl` + `Bridge_LLVM.jl`)
- Primary: Uses Clang.jl for type-aware bindings from headers
- Fallback: Basic ccall wrappers from symbol extraction
- Auto-generates Julia modules with proper type mappings

### Configuration System

All projects use `replibuild.toml` files with these sections:
- `[project]` - Name, root, UUID
- `[discovery]` - Scan results, dependency graphs, include paths
- `[compile]` - Source files, flags, include dirs, link libraries
- `[link]` - Optimization settings
- `[binary]` - Library/executable output settings
- `[symbols]` - Symbol extraction config
- `[wrap]` - Binding generation settings
- `[workflow]` - Build stages to execute
- `[cache]` - Incremental build cache settings
- `[llvm]` - Toolchain configuration

### Multi-Library Workspace Structure

RepliBuild detects workspaces by finding subdirectories with `replibuild.toml`:
```
workspace/
├── lib1/
│   ├── src/
│   └── replibuild.toml
├── lib2/
│   ├── src/
│   └── replibuild.toml
└── app/
    ├── src/
    └── replibuild.toml  # Can depend on lib1, lib2
```

**Workspace Build Process**:
1. `WorkspaceBuilder.discover_workspace()` finds all library targets
2. Parses each `replibuild.toml` to extract `link_libraries` dependencies
3. `compute_build_order()` performs topological sort by levels
4. Each level is built in parallel using `@task` and `schedule()`
5. Built libraries are passed to dependent targets

**Dependency Resolution**: Libraries reference each other by name in `compile.link_libraries`. System libraries (pthread, dl, rt, m) are filtered out.

### Incremental Build System

Build caching is critical for performance:
- `needs_recompile(source, ir_file, cache_enabled)` checks mtime
- IR files cached in `build/` directory
- Only recompiles when source is newer than IR
- Parallel compilation uses `Threads.@threads` for independent files
- Cache hit rates displayed: `⚡ 75% cache hit`

### LLVM Toolchain Discovery

`LLVMEnvironment.jl` discovers LLVM tools in order:
1. Check JLL package (LLVM_full_assert_jll)
2. Check bundled `LLVM/` directory in package
3. Search system PATH

Tools cached in config to avoid repeated searches. Use `BuildBridge.execute(tool_name, args)` to run LLVM commands.

## Common Patterns

### Adding a New Build Stage

Build stages are defined in `workflow.stages` in replibuild.toml. To add a new stage:

1. Add stage logic to `Bridge_LLVM.compile_project()` with conditional:
   ```julia
   if "new_stage" in config.stages
       result = perform_new_stage(config, input)
   end
   ```

2. Update default workflow in `Discovery.generate_config()`:
   ```julia
   "stages" => ["discovery", "compile", "link", "binary", "new_stage", "symbols", "wrap"]
   ```

### Adding a REPL Command

Add to `REPL_API.jl`:
```julia
function rnewcommand(args...)
    # Implementation using RepliBuild.* or Bridge_LLVM.* functions
end
export rnewcommand
```

Then export from main `RepliBuild.jl`:
```julia
export rnewcommand
```

### Working with Dependency Graphs

Dependency graphs are stored as JSON in `.replibuild_cache/dependency_graph.json`:
```julia
# Load graph
dep_graph = ASTWalker.load_dependency_graph_json(path)

# Access compilation order (respects dependencies)
dep_graph.compilation_order  # Vector{String} - sorted files

# Build project using dependency order
cpp_files = filter(f -> endswith(f, ".cpp"), dep_graph.compilation_order)
```

### CMake Project Import

`import_cmake()` parses CMakeLists.txt without running CMake:
- Tokenizes CMake commands
- Extracts targets (add_library, add_executable)
- Resolves dependencies (target_link_libraries)
- Generates replibuild.toml for each target
- Creates multi-library workspace structure

Use for migrating CMake projects to RepliBuild.

## File Organization

**Core Modules** (in `src/`):
- `RepliBuild.jl` - Main entry point, exports public API
- `Bridge_LLVM.jl` - ~1100 lines, main compilation orchestrator
- `WorkspaceBuilder.jl` - Multi-library parallel builds (~387 LOC)
- `Discovery.jl` - Project scanning and analysis (~600 LOC)
- `ASTWalker.jl` - Dependency graph construction
- `ConfigurationManager.jl` - TOML config management (~600 LOC)
- `LLVMEnvironment.jl` - Toolchain discovery
- `BuildBridge.jl` - Tool execution wrapper (~328 LOC, simple & focused)
- `CMakeParser.jl` - CMake import functionality
- `ClangJLBridge.jl` - Clang.jl integration for bindings (~297 LOC)
- `REPL_API.jl` - Interactive convenience commands (~370 LOC)

**External Modules** (referenced but in separate directory):
- `replibuild/ModuleRegistry.jl` - External library resolution
- `replibuild/ModuleTemplateGenerator.jl` - Project scaffolding

**Recently Removed** (2024 cleanup):
The following modules were removed to simplify the codebase:
- `LLVMake.jl` - Redundant with Bridge_LLVM.jl (deleted)
- `JuliaWrapItUp.jl` - Over-engineered wrapper (deleted)
- `ErrorLearning.jl` - Error database system (deleted)
- Daemon system - Background compilation (removed)
- UX helpers, build system delegate (removed)

These features are no longer available - use the streamlined API instead.

## Public API

The main user-facing functions (exported from `RepliBuild`):

- `discover(path; force)` - Analyze project and generate config
- `build(path; parallel, clean_first)` - Build project
- `import_cmake(path; dry_run)` - Import CMake project
- `clean(path)` - Remove build artifacts
- `info(path)` - Show project status

REPL shortcuts (from `REPL_API`):
- `rbuild()`, `rdiscover()`, `rclean()`, `rinfo()`
- `rbuild_fast(sources)` - Quick compilation without discovery
- `rcompile(files...)` - Compile without linking
- `rwrap(lib_path)` - Generate bindings for library
- `rthreads()`, `rcache_status()`

## Key Implementation Details

### Parallel Compilation

Bridge_LLVM uses thread-based parallelism:
```julia
if config.parallel && length(files_to_compile) > 1
    results = Vector{Tuple{String,Bool,Int}}(undef, length(files_to_compile))
    Threads.@threads for i in 1:length(files_to_compile)
        results[i] = compile_single_to_ir(config, files_to_compile[i])
    end
end
```

Always start Julia with `--threads=auto` for parallel builds.

### Workspace Build Levels

`WorkspaceBuilder.compute_build_order()` uses Kahn's algorithm variant:
- Finds all targets with no unbuilt dependencies → Level 1
- Removes built targets, repeat → Level 2, 3, ...
- Each level builds in parallel using `@task`
- Detects circular dependencies

### Symbol Extraction and Binding Generation

```julia
# Extract symbols with nm
symbols = extract_symbols(config, lib_path)

# Generate Julia module
generate_julia_bindings(config, lib_path, symbols, functions)
```

Generated bindings create a Julia module with ccall wrappers for each exported symbol.

## Important Notes

- **Thread Safety**: Compilation tasks must be independent. WorkspaceBuilder ensures this via dependency levels.
- **Path Handling**: All paths in TOML configs should be relative to project root. Use `abspath()` when needed.
- **LLVM Version**: Supports LLVM 18-21 via JLL packages (see Project.toml compat).
- **Git Status**: Many files shown as deleted in current working tree - this is a major refactoring in progress. The core build system (Discovery, Bridge_LLVM, WorkspaceBuilder) is intact.
- **No Tests**: Test suite has been removed. When adding tests, create new `test/runtests.jl`.
